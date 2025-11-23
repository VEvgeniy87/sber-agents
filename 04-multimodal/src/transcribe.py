import logging
import asyncio
import os
import tempfile
from config import config

# Пытаемся импортировать whisper, но если не получается - продолжаем без него
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Whisper successfully imported")
except ImportError as e:
    WHISPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Whisper not available: {e}")
except OSError as e:
    WHISPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"PyTorch DLL error, Whisper not available: {e}")

class VoiceTranscriber:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.whisper_available = WHISPER_AVAILABLE
        
    async def load_model(self):
        """Загружает модель Whisper асинхронно"""
        if self.model_loaded or not self.whisper_available:
            return
            
        try:
            # Загружаем модель в отдельном потоке чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model("base", device="cpu")
            )
            self.model_loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_available = False
    
    async def transcribe_voice(self, voice_file_path: str) -> str:
        """
        Транскрибирует голосовое сообщение с использованием локального Whisper
        """
        if not self.whisper_available:
            return None
            
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # Проверяем что файл существует
            if not os.path.exists(voice_file_path):
                logger.error(f"Voice file not found: {voice_file_path}")
                return "Ошибка: файл не найден"
            
            # Транскрибируем в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(voice_file_path, language="ru")
            )
            
            text = result.get("text", "").strip()
            if text:
                logger.info(f"Transcription successful: {text}")
                return text
            else:
                logger.warning("Transcription returned empty text")
                return "Не удалось распознать речь"
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Ошибка транскрибации: {str(e)}"
    
    async def download_voice_file(self, file_id: str, bot) -> str:
        """
        Скачивает голосовое сообщение из Telegram
        """
        try:
            # Получаем информацию о файле
            file_info = await bot.get_file(file_id)
            file_path = file_info.file_path
            
            # Создаем временный файл
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Скачиваем файл
            await bot.download_file(file_path, temp_file_path)
            logger.info(f"Voice file downloaded to: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error downloading voice file: {e}")
            return None
    
    def cleanup_temp_file(self, file_path: str):
        """Удаляет временный файл"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")

# Создаем экземпляр транскрибера
transcriber = VoiceTranscriber()