#!/usr/bin/env python3
"""
Альтернативный модуль транскрибации без PyTorch
Использует более легковесные решения или внешние API
"""

import asyncio
import logging
import io
import wave
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

async def transcribe_voice_message_alt(file_bytes: io.BytesIO) -> Optional[str]:
    """
    Альтернативная функция транскрибации голосовых сообщений.
    В текущей реализации возвращает тестовый текст с вариациями.
    В реальном использовании можно интегрировать с:
    - Google Speech-to-Text API
    - Yandex SpeechKit
    - Azure Speech Services
    - Другими облачными сервисами
    """
    
    try:
        logger.info("Альтернативная транскрибация голосового сообщения")
        
        # Временная заглушка - возвращаем тестовый текст с вариациями
        # В реальном приложении здесь будет интеграция с API транскрибации
        
        # Пример: анализ аудиофайла для получения базовой информации
        try:
            # Пробуем прочитать WAV файл для получения информации
            file_bytes.seek(0)
            with wave.open(file_bytes, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logger.info(f"Аудио информация: {channels} каналов, {sample_width} байт/сэмпл, {frame_rate} Гц, {n_frames} фреймов")
                
                # Генерируем разные тексты в зависимости от длины аудио
                # Это имитация распознавания разных сообщений
                if n_frames < 500:
                    return "сегодня купил продукты на две тысячи рублей"
                elif n_frames < 1000:
                    return "потратил 500 рублей на продукты"
                elif n_frames < 1500:
                    return "заплатил 1500 рублей за обед в ресторане"
                elif n_frames < 2000:
                    return "получил зарплату 50000 рублей"
                else:
                    return "купил билеты в кино на 800 рублей"
                
        except Exception as e:
            logger.warning(f"Не удалось проанализировать аудиофайл: {e}")
        
        # Возвращаем заглушку для демонстрации (только текст транзакции без префикса)
        return "сегодня купил продукты на две тысячи рублей"
        
    except Exception as e:
        logger.error(f"Ошибка в альтернативной транскрибации: {e}")
        return None

async def test_transcription():
    """Тестирование альтернативной транскрибации"""
    # Создаем тестовый аудиофайл
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    buffer.seek(0)
    
    result = await transcribe_voice_message_alt(buffer)
    print(f"Результат транскрибации: {result}")

if __name__ == "__main__":
    asyncio.run(test_transcription())