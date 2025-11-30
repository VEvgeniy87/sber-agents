import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
from langchain_community.document_loaders import JSONLoader

logger = logging.getLogger(__name__)

def load_json_documents(json_file_path: str) -> list:
    """
    Загрузка документов из JSON файла с вопросами-ответами
    Каждая пара Q&A становится отдельным чанком
    """
    from pathlib import Path
    
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    # JSONLoader с jq_schema для извлечения full_text из каждого элемента массива
    loader = JSONLoader(
        file_path=str(json_path),
        jq_schema='.[].full_text',  # Извлекаем full_text из каждого элемента
        text_content=False
    )
    
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
    return documents

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
        # Улучшенный TF-IDF векторизатор для русского языка
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=None,  # Убираем английские стоп-слова для русского текста
            ngram_range=(1, 2),  # Добавляем биграммы для лучшего поиска
            min_df=1,  # Минимальная частота термина
            max_df=0.9  # Максимальная частота термина
        )
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
    
    def keyword_search(self, query, k=3):
        """Простой поиск по ключевым словам для русского языка"""
        if not self.documents:
            return []
        
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            score = 0
            
            # Подсчитываем совпадения ключевых слов
            for word in query_words:
                if len(word) > 3:  # Игнорируем короткие слова
                    score += content.count(word) * len(word)  # Более длинные слова имеют больший вес
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Сортируем по убыванию релевантности
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Возвращаем топ-k документов
        return [doc for _, doc in scored_docs[:k]]
    
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
            
            def __call__(self, query):
                """Поддержка вызова как функции для совместимости с LangChain цепями"""
                return self.invoke(query)
        
        return LocalRetriever(self, k)

def create_vector_store(chunks: list):
    """Создание векторного хранилища с локальными TF-IDF эмбеддингами"""
    # Используем только локальные TF-IDF эмбеддинги для быстрого запуска
    vector_store = LocalVectorStore()
    vector_store.add_documents(chunks)
    logger.info(f"Created local TF-IDF vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)"""
    logger.info("Starting full reindexing...")
    
    try:
        # 1. Загружаем и обрабатываем PDF документы
        pdf_pages = load_pdf_documents(config.DATA_DIR)
        if not pdf_pages:
            logger.warning("No PDF documents found to index")
        
        pdf_chunks = split_documents(pdf_pages) if pdf_pages else []
        
        # 2. Загружаем JSON с вопросами-ответами
        json_file = f"{config.DATA_DIR}/sberbank_help_documents.json"
        json_chunks = load_json_documents(json_file)
        
        # 3. Объединяем все чанки
        all_chunks = pdf_chunks + json_chunks
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_chunks)})")
            
        # 4. Создаём векторное хранилище
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

