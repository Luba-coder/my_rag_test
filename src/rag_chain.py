# src/rag_chain.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os

class SimpleRAG:
    def __init__(self, vectorstore_path: str = "vectorstore"):
        if not os.path.exists(vectorstore_path):
            raise RuntimeError("Векторное хранилище не найдено. Сначала запустите этап создания.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    def retrieve_context(self, query: str, k: int = 2) -> str:
        """
        Ищет релевантные фрагменты из syllabus.txt по запросу.
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def generate_answer(self, query: str) -> str:
        """
        Эмулирует генерацию ответа: просто возвращает найденный контекст.
        В реальной системе здесь стояла бы LLM.
        """
        context = self.retrieve_context(query)
        # В реальном RAG здесь был бы промпт вроде:
        # "Ответь на вопрос, используя только следующий контекст: {context}. Вопрос: {query}"
        # Но мы пока просто возвращаем контекст — этого достаточно для тестирования!
        return f"Найденный контекст:\n{context}"