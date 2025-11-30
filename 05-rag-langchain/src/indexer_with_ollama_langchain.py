import os
import logging
from typing import List
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaVectorStore:
    """Векторное хранилище с использованием Ollama через LangChain"""
    
    def __init__(self, model_name: str = "aroxima/multilingual-e5-large-instruct:latest"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self.documents: List[Document] = []
        
    def add_documents(self, documents: List[Document]):
        """Добавить документы в векторное хранилище"""
        self.documents.extend(documents)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Поиск похожих документов по запросу"""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def keyword_search(self, query: str, k: int = 3) -> List[Document]:
        """Поиск по ключевым словам (fallback метод)"""
        if not self.documents:
            return []
            
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content_words = set(doc.page_content.lower().split())
            score = len(query_words.intersection(content_words))
            scored_docs.append((score, doc))
        
        # Сортируем по релевантности и берем топ-k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k] if score > 0]
    
    def as_retriever(self, search_type: str = "similarity", **kwargs):
        """Создает retriever для LangChain"""
        if self.vector_store is None:
            from langchain_core.retrievers import BaseRetriever
            
            class EmptyRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str) -> List[Document]:
                    return []
            
            return EmptyRetriever()
        
        if search_type == "similarity":
            return self.vector_store.as_retriever(search_type="similarity", search_kwargs=kwargs)
        elif search_type == "keyword":
            from langchain_core.retrievers import BaseRetriever
            
            class KeywordRetriever(BaseRetriever):
                def __init__(self, vector_store, **kwargs):
                    self.vector_store = vector_store
                    self.kwargs = kwargs
                    
                def _get_relevant_documents(self, query: str) -> List[Document]:
                    return self.vector_store.keyword_search(query, **self.kwargs)
            
            return KeywordRetriever(self, **kwargs)
        else:
            # Комбинированный поиск
            return self.vector_store.as_retriever(search_type="similarity", search_kwargs=kwargs)

def load_pdf_documents(data_dir: str = "data") -> List[Document]:
    """Загрузить документы из PDF файлов"""
    pdf_documents = []
    data_path = Path(data_dir)
    
    for pdf_file in data_path.glob("*.pdf"):
        try:
            logger.info(f"Loading {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            pdf_documents.extend(documents)
            logger.info(f"Loaded {pdf_file.name}: {len(documents)} pages")
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")
    
    return pdf_documents

def load_json_documents(data_dir: str = "data") -> List[Document]:
    """Загрузить Q&A пары из JSON файлов"""
    json_documents = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        try:
            logger.info(f"Loading Q&A pairs from {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Предполагаем структуру с вопросами и ответами
            for item in data:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    # Создаем документ с вопросом и ответом
                    content = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
                    metadata = {
                        'source': str(json_file),
                        'type': 'qna',
                        'question': item['question']
                    }
                    json_documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Loaded {json_file.name}: {len(json_documents)} Q&A pairs")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    return json_documents

def reindex_all(data_dir: str = "data", chunk_size: int = 1500, chunk_overlap: int = 150) -> OllamaVectorStore:
    """Полная переиндексация всех документов с использованием Ollama через LangChain"""
    logger.info("Starting full reindexing with Ollama embeddings (LangChain)...")
    
    # Загружаем документы
    pdf_docs = load_pdf_documents(data_dir)
    json_docs = load_json_documents(data_dir)
    
    # Разбиваем на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    # Разбиваем PDF документы
    pdf_chunks = text_splitter.split_documents(pdf_docs)
    logger.info(f"Split into {len(pdf_chunks)} PDF chunks")
    
    # JSON документы уже структурированы, не разбиваем
    all_chunks = pdf_chunks + json_docs
    logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_docs)})")
    
    # Создаем векторное хранилище с Ollama
    vector_store = OllamaVectorStore()
    vector_store.add_documents(all_chunks)
    
    logger.info("Reindexing completed successfully")
    return vector_store

if __name__ == "__main__":
    # Тестирование
    vector_store = reindex_all()
    print(f"Vector store created with {len(vector_store.documents)} documents")
    
    # Тестовый поиск
    test_query = "Как заказать карту?"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"Results for '{test_query}':")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:200]}...")