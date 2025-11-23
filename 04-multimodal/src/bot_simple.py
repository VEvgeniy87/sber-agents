#!/usr/bin/env python3
"""
Простая версия бота без транскрибации для тестирования
"""

import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import config
from handlers import router

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def main():
    """Запуск бота"""
    logger = logging.getLogger(__name__)
    
    # Создаем бота с настройками по умолчанию
    bot = Bot(
        token=config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    
    # Создаем диспетчер
    dp = Dispatcher()
    
    # Подключаем роутер
    dp.include_router(router)
    
    logger.info("🤖 Бот запускается...")
    
    try:
        # Запускаем бота
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())