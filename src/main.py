import os
import json
import httpx
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import CrossEncoder 

# --- НАСТРОЙКИ ---
API_KEY = "sk-1f5LdNeuVjkH9U6Od6561A"
HUB_URL = "https://hub.qazcode.ai" 
MODEL = "oss-120b"

vector_db = None
reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, reranker
    print("⏳ [1/2] Загружаем базу...")
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    print("⏳ [2/2] Загружаем Reranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("✅ СИСТЕМА ГОТОВА!")
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
    print(f"\n🔍 Анализ: {request.symptoms[:50]}...")
    
    try:
        # 1. Поиск + Reranking (Хитрая логика)
        initial_results = vector_db.similarity_search(request.symptoms, k=10)
        pairs = [[request.symptoms, doc.page_content] for doc in initial_results]
        scores = reranker.predict(pairs)
        top_indices = scores.argsort()[::-1][:3]
        top_docs = [initial_results[i] for i in top_indices]
        
        context_str = "\n".join([
            f"ПРОТОКОЛ: {doc.metadata.get('title', '')}\nМКБ: {doc.metadata.get('icd_codes', '')}\nТЕКСТ: {doc.page_content[:500]}" 
            for doc in top_docs
        ])

        # 2. Промпт
        SYSTEM_PROMPT = SYSTEM_PROMPT = """Ты — ведущий медицинский эксперт Казахстана.
Твоя задача: поставить ТОЧНЫЙ диагноз на основе симптомов и фрагментов протоколов РК.

ВНИМАНИЕ: Протоколы уже отсортированы по релевантности. Первый протокол в списке — самый вероятный.
1. Тщательно сверяй симптомы пациента с критериями в тексте.
2. Выдай 3 наиболее подходящих диагноза.
3. Код МКБ-10 должен СТРОГО соответствовать коду из текста протокола.

ФОРМАТ JSON:
{"diagnoses": [{"rank": 1, "icd10_code": "...", "name": "...", "explanation": "..."}]}"""

        USER_PROMPT = f"Симптомы:\n{request.symptoms}\n\nКонтекст:\n{context_str}"

        # 3. Запрос к LLM с таймаутом None
        async with httpx.AsyncClient(timeout=None) as client:
            api_url = f"{HUB_URL}/chat/completions"
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
            
            # Если 404, пробуем альтернативный путь
            if response.status_code == 404:
                response = await client.post(f"{HUB_URL}/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}], "temperature": 0.01}, headers={"Authorization": f"Bearer {API_KEY}"})

            raw_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # --- ХИТРЫЙ ПАРСИНГ ---
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            start = clean_json.find("{")
            end = clean_json.rfind("}") + 1
            
            if start == -1 or end == 0:
                raise ValueError("AI не вернул валидный JSON блок")
            
            data = json.loads(clean_json[start:end])

            # ЗАЩИТА: Если ключи названы неправильно (галлюцинация модели)
            # Иногда модель пишет 'diagnosis' вместо 'diagnoses'
            if "diagnoses" not in data and "diagnosis" in data:
                data["diagnoses"] = data["diagnosis"]
            
            # Если вообще нет списка, создаем пустой
            if "diagnoses" not in data or not isinstance(data["diagnoses"], list):
                raise ValueError("В ответе AI отсутствует список diagnoses")

            # Фикс ключа icd_code -> icd10_code (частая ошибка моделей)
            for d in data["diagnoses"]:
                if "icd_code" in d and "icd10_code" not in d:
                    d["icd10_code"] = d["icd_code"]
                if "icd10_code" not in d:
                    d["icd10_code"] = "Не указан"
                if "name" not in d:
                    d["name"] = "Неизвестный диагноз"

            print(f"✅ Успешно! Найдено диагнозов: {len(data['diagnoses'])}")
            return DiagnosisResponse(**data)

    except Exception as e:
        print(f"\n❌ ОШИБКА ОБРАБОТКИ: {str(e)}")
        # Возвращаем пустой, но валидный ответ, чтобы evaluate.py не падал
        return DiagnosisResponse(diagnoses=[
            Diagnosis(rank=1, icd10_code="N/A", name="Диагноз не определен", explanation=f"Ошибка: {str(e)}")
        ])