"""Основной модуль Telegram бота."""
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from .config import load_config
from .llm_client import LLMClient


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def start_command(message: types.Message):
    """Обработчик команды /start."""
    await message.answer(
        "👋 Привет! Я кулинарный помощник. "
        "Задавайте вопросы о готовке, рецептах и кухонных хитростях!"
    )


async def help_command(message: types.Message):
    """Обработчик команды /help."""
    await message.answer(
        "🍳 Я помогу вам с:\n"
        "• Рецептами и техниками приготовления\n"
        "• Заменой ингредиентов\n"
        "• Советами по кухонному оборудованию\n"
        "• Планированием питания\n\n"
        "Просто напишите ваш вопрос!"
    )


async def handle_text_message(message: types.Message):
    """Обработчик текстовых сообщений с использованием LLM."""
    try:
        # Показать статус обработки
        await message.answer("🍳 Думаю над ответом...")
        
        # Загрузить конфигурацию и создать LLM клиент
        config = load_config()
        client = LLMClient(api_key=config["api_key"], model=config["llm_model"])
        
        # Получить ответ от LLM
        response = await client.get_response(message.text)
        
        # Отправить ответ пользователю
        await message.answer(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        
        # Более понятное сообщение об ошибке для пользователя
        error_message = str(e)
        if "429" in error_message:
            user_friendly_error = (
                "🍳 Извините, сейчас сервис перегружен. "
                "Пожалуйста, попробуйте задать вопрос через несколько минут. "
                "Это временная проблема с бесплатными моделями."
            )
        elif "403" in error_message:
            user_friendly_error = (
                "🍳 Извините, модель временно недоступна в вашем регионе. "
                "Попробуйте позже или используйте другую модель в настройках."
            )
        else:
            user_friendly_error = (
                "🍳 Извините, произошла техническая ошибка. "
                "Пожалуйста, попробуйте позже или задайте другой вопрос."
            )
        
        await message.answer(user_friendly_error)


async def main():
    """Основная функция запуска бота."""
    try:
        # Загрузка конфигурации
        config = load_config()
        logger.info("Конфигурация загружена успешно")
        
        # Инициализация бота и диспетчера
        bot = Bot(token=config["bot_token"])
        dp = Dispatcher()
        
        # Регистрация обработчиков команд
        dp.message.register(start_command, Command("start"))
        dp.message.register(help_command, Command("help"))
        dp.message.register(handle_text_message)
        
        logger.info(f"Бот {config['bot_name']} запущен и готов к работе")
        await dp.start_polling(bot)
        
    except ValueError as e:
        logger.error(f"Ошибка конфигурации: {e}")
        print(f"Ошибка конфигурации: {e}")
        print("Создайте .env файл с TELEGRAM_BOT_TOKEN и OPENROUTER_API_KEY")
        print("Пример файла .env.example можно скопировать и заполнить")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())