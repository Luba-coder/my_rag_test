# src/vectorstore.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .loader import load_and_split_syllabus
import os

def create_vectorstore(syllabus_path: str, persist_dir: str = "vectorstore"):
    """
    Создаёт векторное хранилище на основе syllabus.txt и сохраняет его на диск.
    """
    chunks = load_and_split_syllabus(syllabus_path)

    # Используем бесплатную локальную модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Создаём FAISS хранилище
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Сохраняем на диск
    vectorstore.save_local(persist_dir)
    print(f"Векторное хранилище сохранено в: {persist_dir}")
    return vectorstore