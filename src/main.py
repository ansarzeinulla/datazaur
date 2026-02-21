import os
import json
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# --- НАСТРОЙКИ ---
API_KEY = "sk-kDGHTZAOX-jQcN8VXxQucg"  # Вставь реальный
HUB_URL = "https://hub.qazcode.ai" # Вставь как в ноутбуке. Если там с https://, оставь с https://
MODEL = "oss-120b"

vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    print("⏳ Загружаем векторную базу ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    print("✅ База готова к поиску!")
    yield

app = FastAPI(lifespan=lifespan)

# --- МОДЕЛИ ДАННЫХ ---
class SymptomRequest(BaseModel):
    symptoms: str

class Diagnosis(BaseModel):
    rank: int
    icd10_code: str
    name: str
    explanation: str

class DiagnosisResponse(BaseModel):
    diagnoses: list[Diagnosis]


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
        
# --- РОУТ ---
@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: SymptomRequest):
    print(f"\n🔍 Ищем диагноз для: {request.symptoms[:50]}...")
    
    # 1. Поиск в базе
    results = vector_db.similarity_search(request.symptoms, k=3)
    context_str = "\n\n".join(
        f"Протокол: {doc.metadata.get('title', '')}\nКоды МКБ: {doc.metadata.get('icd_codes', '')}\nТекст: {doc.page_content}"
        for doc in results
    )

    # 2. Промпт
    SYSTEM_PROMPT = """Ты — клинический ассистент. 
На основе симптомов и контекста протоколов РК, выдай 3 вероятных диагноза.
Верни СТРОГО JSON:
{
  "diagnoses": [
    {"rank": 1, "icd10_code": "КОД_ИЗ_КОНТЕКСТА", "name": "название", "explanation": "обоснование"},
    {"rank": 2, "icd10_code": "...", "name": "...", "explanation": "..."},
    {"rank": 3, "icd10_code": "...", "name": "...", "explanation": "..."}
  ]
}"""

    USER_PROMPT = f"Симптомы:\n{request.symptoms}\n\nКонтекст протоколов:\n{context_str}"

    # Формируем правильный URL (как просят организаторы на 4 странице PDF)
    api_url = f"https://{HUB_URL}/chat/completions" if "http" not in HUB_URL else f"{HUB_URL}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        "temperature": 0.1
    }

    # 3. Прямой HTTP запрос (без магии библиотеки openai)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, headers=headers, timeout=60.0)
            
            # Если сервер выдал ошибку (например 401 Неверный ключ)
            if response.status_code != 200:
                print(f"❌ Ошибка сервера QazCode: {response.status_code} - {response.text}")
                raise Exception(f"HTTP {response.status_code}")

            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
            print("✅ Ответ от LLM получен!")
            
            # Парсим JSON
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            parsed_json = json.loads(raw_text)
            return DiagnosisResponse(**parsed_json)
            
    except Exception as e:
        print(f"❌ ОШИБКА В БЛОКЕ LLM: {str(e)}")
        # Возвращаем заглушку, чтобы evaluate.py не сломался, пока мы дебажим
        return DiagnosisResponse(
            diagnoses=[
                Diagnosis(rank=1, icd10_code="000.0", name="Ошибка", explanation=str(e))
            ]
        )