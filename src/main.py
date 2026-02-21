import os
import json
import httpx
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- НАСТРОЙКИ ---
API_KEY = "sk-kDGHTZAOX-jQcN8VXxQucg"
# Пробуем базовый URL без лишних путей
HUB_URL = "https://hub.qazcode.ai" 
MODEL = "oss-120b"

vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    print("⏳ Загружаем базу...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
        vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        print("✅ База готова!")
    except Exception as e:
        print(f"❌ Ошибка базы: {e}")
    yield

app = FastAPI(lifespan=lifespan)

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

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: SymptomRequest):
    print(f"\n🔍 Обработка запроса: {request.symptoms[:50]}...")
    
    try:
        # 1. Поиск (сокращаем k до 3, чтобы модель быстрее читала)
        results = vector_db.similarity_search(request.symptoms, k=3)
        context_str = "\n".join([f"ПРОТОКОЛ: {doc.page_content[:500]}" for doc in results])

        # 2. Промпт (максимально лаконичный)
        SYSTEM_PROMPT = "Ты врач. Проанализируй симптомы по протоколам и верни ТОЛЬКО JSON: {'diagnoses': [{'rank': 1, 'icd10_code': '...', 'name': '...', 'explanation': '...'}]}"
        USER_PROMPT = f"Симптомы: {request.symptoms}\nКонтекст: {context_str}"

        # 3. Запрос с ОГРОМНЫМ таймаутом
        # Пробуем стандартный эндпоинт чата
        api_url = f"{HUB_URL}/chat/completions"
        
        # Используем timeout=None, чтобы ждать столько, сколько нужно
        async with httpx.AsyncClient(timeout=None) as client:
            print("📡 Отправка запроса в QazCode (может занять 1-3 минуты)...")
            response = await client.post(
                api_url, 
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT}
                    ],
                    "temperature": 0.01
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code != 200:
                print(f"❌ Ошибка API ({response.status_code}): {response.text}")
                if response.status_code == 404:
                    print("🔄 Пробую альтернативный URL с /v1...")
                    response = await client.post(
                        f"{HUB_URL}/v1/chat/completions",
                        json={
                            "model": MODEL,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": USER_PROMPT}
                            ],
                            "temperature": 0.01
                        },
                        headers={"Authorization": f"Bearer {API_KEY}"}
                    )

            raw_text = response.json()["choices"][0]["message"]["content"].strip()
            print(f"✅ Ответ получен! Длина: {len(raw_text)} символов.")

            # Очистка от мусора
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            
            # Находим границы JSON если модель добавила лишний текст
            start = clean_json.find("{")
            end = clean_json.rfind("}") + 1
            if start != -1 and end != 0:
                clean_json = clean_json[start:end]

            data = json.loads(clean_json)
            return DiagnosisResponse(**data)

    except Exception as e:
        print("\n❌ ОШИБКА:")
        traceback.print_exc()
        return DiagnosisResponse(diagnoses=[
            Diagnosis(rank=1, icd10_code="TIMEOUT", name="Превышено время ожидания", explanation="Модель oss-120b отвечает слишком долго. Попробуйте еще раз.")
        ])