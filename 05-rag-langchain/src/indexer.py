import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import config

logger = logging.getLogger(__name__)

def load_pdf_documents(data_dir: str) -> list:
    """Загрузка всех PDF документов из директории"""
    pages = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory {data_dir} does not exist")
        return pages
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages.extend(loader.load())
        logger.info(f"Loaded {pdf_file.name}")
    
    return pages

def split_documents(pages: list) -> list:
    """Разбиение документов с учетом структуры"""
    # Сепараторы для банковских документов
    # Пробуем разбивать по: двойным переносам строк, одинарным, пробелам
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n\n",    # Тройной перенос - обычно разделы
            "\n\n",      # Двойной перенос - параграфы
            "\n",        # Одинарный перенос
            ". ",        # Конец предложения
            " ",         # Пробелы
            ""           # Символы
        ],
        keep_separator=True  # Сохраняем разделители для контекста
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

class LocalVectorStore:
    """Простое локальное векторное хранилище на основе TF-IDF"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.embeddings = None
        
    def add_documents(self, documents):
        """Добавление документов в хранилище"""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # Создаем TF-IDF матрицу
        self.embeddings = self.vectorizer.fit_transform(texts)
        logger.info(f"Created local vector store with {len(documents)} documents")
        
    def similarity_search(self, query, k=3):
        """Поиск похожих документов"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
            
        # Преобразуем запрос в TF-IDF вектор
        query_vec = self.vectorizer.transform([query])
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        
        # Получаем топ-k наиболее похожих документов
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Только документы с положительным сходством
                results.append(self.documents[idx])
                
        return results
    
    def as_retriever(self, search_kwargs=None):
        """Создание retriever для совместимости с LangChain"""
        if search_kwargs is None:
            search_kwargs = {}
        
        k = search_kwargs.get('k', 3)
        
        from langchain_core.runnables import Runnable
        
        class LocalRetriever(Runnable):
            def __init__(self, vector_store, k):
                super().__init__()
                self.vector_store = vector_store
                self.k = k
            
            def invoke(self, query, config=None):
                return self.vector_store.similarity_search(query, self.k)
            
            async def ainvoke(self, query, config=None):
                return self.invoke(query, config)
        
        return LocalRetriever(self, k)

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    try:
        # Пробуем использовать OpenRouter эмбеддинги
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL
        )
        vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        logger.info(f"Created OpenAI vector store with {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        logger.warning(f"Failed to use OpenAI embeddings: {e}, falling back to local TF-IDF")
        # Fallback на локальные TF-IDF эмбеддинги
        vector_store = LocalVectorStore()
        vector_store.add_documents(chunks)
        return vector_store

async def reindex_all():
    """Полная переиндексация всех документов"""
    logger.info("Starting full reindexing...")
    
    try:
        pages = load_pdf_documents(config.DATA_DIR)
        if not pages:
            logger.warning("No documents found to index")
            return None
        
        chunks = split_documents(pages)
        if not chunks:
            logger.warning("No chunks created after splitting")
            return None
            
        vector_store = create_vector_store(chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

