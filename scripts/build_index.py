import json
import os
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm # Прогресс-бар нужен, чтобы не скучать

CORPUS_DIR = Path("data/corpus")
DB_DIR = "chroma_db"
BATCH_SIZE = 200 # Пишем пачками, чтобы не забить память

def build_vector_db():
    # 0. Чистим старую базу
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    print("⏳ Загружаем модель (rubert-tiny2)...")
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    
    # Используем 500/100 (лучше для точности, чем 1000)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    metadatas = []

    print(f"📂 Сканируем {CORPUS_DIR}...")
    json_files = list(CORPUS_DIR.glob("*.json"))

    # 1. Читаем файлы (в одном потоке, зато стабильно)
    for file_path in tqdm(json_files, desc="Чтение файлов"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            items = data if isinstance(data, list) else [data]
                
            for item in items:
                text = item.get("text", "")
                if not text: continue
                    
                # Данные для контекста
                title = item.get('title', 'Без названия')
                codes = ", ".join(item.get('icd_codes', []))
                
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    # ХАК ДЛЯ ПОБЕДЫ: Вшиваем контекст прямо в текст
                    enriched_text = f"БОЛЕЗНЬ: {title}. КОД МКБ: {codes}. ТЕКСТ: {chunk}"
                    
                    documents.append(enriched_text)
                    metadatas.append({
                        "protocol_id": item.get("protocol_id", ""),
                        "title": title,
                        "icd_codes": codes
                    })
        except Exception as e:
            print(f"Ошибка с файлом {file_path}: {e}")

    total_chunks = len(documents)
    print(f"🧩 Готово к записи: {total_chunks} чанков.")

    # 2. Пишем в базу
    print("🧠 Создаем базу ChromaDB...")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Пишем батчами (самый надежный способ)
    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Векторизация"):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]
        vector_db.add_texts(texts=batch_docs, metadatas=batch_meta)
    
    print(f"✅ База готова: {DB_DIR}")

if __name__ == "__main__":
    build_vector_db()