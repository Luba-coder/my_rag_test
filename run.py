# run.py

import os
from src.vectorstore import create_vectorstore
from src.rag_chain import SimpleRAG

SYLLABUS_PATH = "data/syllabus.txt"
VECTORSTORE_PATH = "vectorstore"

def main():
    # Шаг 1: Создаём векторное хранилище, если его ещё нет
    if not os.path.exists(VECTORSTORE_PATH):
        print("Создаю векторное хранилище...")
        create_vectorstore(SYLLABUS_PATH, VECTORSTORE_PATH)
    else:
        print("Векторное хранилище уже существует — загружаю.")

    # Шаг 2: Инициализируем RAG
    rag = SimpleRAG(VECTORSTORE_PATH)

    # Шаг 3: Интерактивный режим
    print("\n✅ RAG готов к работе! Задавай вопросы по силлабусу QA.")
    print("Напиши 'exit', чтобы выйти.\n")

    while True:
        query = input("Вопрос: ").strip()
        if query.lower() == "exit":
            break
        answer = rag.generate_answer(query)
        print(f"\nОтвет:\n{answer}\n{'-'*50}\n")

if __name__ == "__main__":
    main()