#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы транскрибации голосовых сообщений
"""

import asyncio
import logging
from src.transcribe import transcribe_voice_message

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_transcription():
    """Тестирует работу транскрибации"""
    print("🧪 Тестирование транскрибации голосовых сообщений...")
    
    # Создаем тестовый файл с голосовым сообщением
    # Для теста создадим простой WAV файл с синусоидальным сигналом
    # (в реальном использовании это будет файл из Telegram)
    
    try:
        # Импортируем необходимые библиотеки для создания тестового аудио
        import numpy as np
        import wave
        import io
        
        # Создаем простой синусоидальный сигнал (1 секунда, 440 Гц)
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Конвертируем в 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Создаем WAV файл в памяти
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        
        print("📝 Тестовый аудиофайл создан")
        
        # Пробуем транскрибировать
        print("🎤 Запуск транскрибации...")
        transcribed_text = await transcribe_voice_message(buffer)
        
        if transcribed_text:
            print(f"✅ Транскрибация успешна: '{transcribed_text}'")
        else:
            print("❌ Транскрибация не удалась")
            
    except ImportError as e:
        print(f"⚠️ Не удалось импортировать необходимые библиотеки: {e}")
        print("Убедитесь, что установлены numpy и wave")
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_transcription())