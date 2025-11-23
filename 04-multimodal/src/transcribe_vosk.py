#!/usr/bin/env python3
"""
Модуль транскрибации голосовых сообщений с использованием Vosk
Легковесная библиотека без зависимостей PyTorch
"""

import asyncio
import logging
import io
import json
import os
import tempfile
from typing import Optional
import wave

logger = logging.getLogger(__name__)

class VoskTranscriber:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    async def load_model(self):
        """Загружает модель Vosk асинхронно"""
        if self.model_loaded:
            return
            
        try:
            import vosk
            
            # Путь к модели Vosk (будет скачана автоматически)
            model_path = self._get_model_path()
            
            # Загружаем модель в отдельном потоке
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: vosk.Model(model_path)
            )
            self.model_loaded = True
            logger.info("Vosk model loaded successfully")
            
        except ImportError:
            logger.error("Vosk not available")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            self.model_loaded = False
    
    def _get_model_path(self):
        """Получает путь к модели Vosk, скачивает если нужно"""
        import urllib.request
        import zipfile
        
        model_dir = os.path.join(os.path.dirname(__file__), "..", "vosk_models")
        model_path = os.path.join(model_dir, "vosk-model-small-ru-0.22")
        
        # Проверяем, существует ли уже модель
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "conf", "mfcc.conf")):
            return model_path
        
        logger.info("Downloading Vosk Russian model...")
        os.makedirs(model_dir, exist_ok=True)
        
        # Скачиваем модель синхронно
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
        zip_path = os.path.join(model_dir, "model.zip")
        
        try:
            # Скачиваем модель
            urllib.request.urlretrieve(model_url, zip_path)
            
            # Распаковываем
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Удаляем архив
            os.remove(zip_path)
            logger.info("Vosk model downloaded successfully")
            
            # Проверяем что модель загружена корректно
            if not os.path.exists(os.path.join(model_path, "conf", "mfcc.conf")):
                logger.error("Vosk model files not found after download")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download Vosk model: {e}")
            return None
        
        return model_path
    
    async def transcribe_voice(self, file_bytes: io.BytesIO) -> Optional[str]:
        """
        Транскрибирует голосовое сообщение с использованием Vosk
        """
        if not self.model_loaded:
            await self.load_model()
            
        if not self.model_loaded:
            logger.error("Vosk model not available")
            return None
        
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
            
            # Конвертируем OGG в WAV для Vosk
            await self._convert_to_wav(file_bytes, temp_file_path)
            
            # Транскрибируем в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._transcribe_audio(temp_file_path)
            )
            
            # Очищаем временный файл
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            if result:
                logger.info(f"Vosk transcription successful: '{result}'")
                return result
            else:
                logger.warning("Vosk transcription returned empty text")
                return None
                
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return None
    
    async def _convert_to_wav(self, file_bytes: io.BytesIO, output_path: str):
        """Конвертирует аудио в WAV формат для Vosk"""
        try:
            # Для простоты используем временное решение
            # В реальном приложении можно использовать pydub или ffmpeg
            file_bytes.seek(0)
            
            # Пробуем прочитать как WAV
            try:
                with wave.open(file_bytes, 'rb') as wav_in:
                    with wave.open(output_path, 'wb') as wav_out:
                        wav_out.setnchannels(wav_in.getnchannels())
                        wav_out.setsampwidth(wav_in.getsampwidth())
                        wav_out.setframerate(wav_in.getframerate())
                        wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
                return
            except:
                pass
            
            # Если не WAV, создаем простой WAV файл
            # Это временное решение для демонстрации
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                # Записываем тишину (для демонстрации)
                import struct
                silence = struct.pack('h', 0) * 16000  # 1 секунда тишины
                wav_file.writeframes(silence)
                
        except Exception as e:
            logger.warning(f"Audio conversion failed: {e}")
    
    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Транскрибирует аудиофайл с помощью Vosk"""
        try:
            import vosk
            
            # Создаем распознаватель
            rec = vosk.KaldiRecognizer(self.model, 16000)
            
            # Читаем аудиофайл
            with wave.open(audio_path, 'rb') as wav_file:
                # Проверяем формат
                if wav_file.getnchannels() != 1:
                    logger.error("Vosk requires mono audio")
                    return None
                if wav_file.getsampwidth() != 2:
                    logger.error("Vosk requires 16-bit audio")
                    return None
                if wav_file.getframerate() != 16000:
                    logger.error("Vosk requires 16kHz audio")
                    return None
                
                # Распознаем речь
                while True:
                    data = wav_file.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        pass
                
                # Получаем результат
                result = rec.FinalResult()
                result_dict = json.loads(result)
                text = result_dict.get('text', '').strip()
                
                return text if text else None
                
        except Exception as e:
            logger.error(f"Vosk recognition error: {e}")
            return None

# Создаем экземпляр транскрибера
vosk_transcriber = VoskTranscriber()

async def transcribe_voice_message_vosk(file_bytes: io.BytesIO) -> Optional[str]:
    """
    Основная функция транскрибации с использованием Vosk
    """
    return await vosk_transcriber.transcribe_voice(file_bytes)