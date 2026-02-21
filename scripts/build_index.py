import json
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = Path("data/corpus") # Убедись, что тут лежит твой .json файл
DB_DIR = "chroma_db"

def build_vector_db():
    print("⏳ Загружаем модель эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    metadatas = []

    print(f"📂 Читаем JSON файлы из {CORPUS_DIR}...")
    json_files = list(CORPUS_DIR.glob("*.json"))
    
    if not json_files:
        print("❌ Ошибка: Файлы не найдены! Проверь путь CORPUS_DIR.")
        return

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # ВАЖНОЕ ИСПРАВЛЕНИЕ: если это список, проходимся по каждому элементу
            if isinstance(data, list):
                items = data
            else:
                items = [data] # На всякий случай, если попадется одиночный объект
                
            for item in items:
                text = item.get("text", "")
                if not text:
                    continue
                    
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    documents.append(chunk)
                    metadatas.append({
                        "protocol_id": item.get("protocol_id", ""),
                        "title": item.get("title", ""),
                        "source_file": item.get("source_file", ""),
                        "icd_codes": ", ".join(item.get("icd_codes", []))
                    })

    print(f"🧩 Подготовлено {len(documents)} кусков текста.")
    print("🧠 Векторизуем и сохраняем в ChromaDB (это может занять пару минут)...")
    
    vector_db = Chroma.from_texts(
        texts=documents,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"✅ База успешно создана и сохранена в папку: ./{DB_DIR}/")

if __name__ == "__main__":
    build_vector_db()