import os
import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingVectorStore:
    """Локальное векторное хранилище с использованием SentenceTransformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def add_documents(self, documents: List[Document]):
        """Добавить документы и вычислить эмбеддинги"""
        self.documents.extend(documents)
        
        # Вычисляем эмбеддинги для всех документов
        texts = [doc.page_content for doc in documents]
        new_embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Поиск похожих документов по запросу"""
        if not self.documents:
            return []
            
        # Эмбеддинг для запроса
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Получаем топ-k документов
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
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
        from langchain_core.retrievers import BaseRetriever
        
        class LocalRetriever(BaseRetriever):
            def __init__(self, vector_store, search_type, **kwargs):
                self.vector_store = vector_store
                self.search_type = search_type
                self.kwargs = kwargs
                
            def _get_relevant_documents(self, query: str) -> List[Document]:
                if self.search_type == "similarity":
                    return self.vector_store.similarity_search(query, **self.kwargs)
                elif self.search_type == "keyword":
                    return self.vector_store.keyword_search(query, **self.kwargs)
                else:
                    # Пробуем оба метода и объединяем результаты
                    similarity_results = self.vector_store.similarity_search(query, **self.kwargs)
                    keyword_results = self.vector_store.keyword_search(query, **self.kwargs)
                    
                    # Объединяем и удаляем дубликаты
                    combined = similarity_results + keyword_results
                    seen = set()
                    unique_results = []
                    for doc in combined:
                        if doc.page_content not in seen:
                            seen.add(doc.page_content)
                            unique_results.append(doc)
                    
                    return unique_results[:kwargs.get('k', 3)]
        
        return LocalRetriever(self, search_type, **kwargs)

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

def reindex_all(data_dir: str = "data", chunk_size: int = 1500, chunk_overlap: int = 150) -> LocalEmbeddingVectorStore:
    """Полная переиндексация всех документов"""
    logger.info("Starting full reindexing with local embeddings...")
    
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
    
    # Создаем векторное хранилище
    vector_store = LocalEmbeddingVectorStore()
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