"""Модуль загрузки конфигурации проекта."""
import os
from typing import Dict


def load_config() -> Dict[str, str]:
    """Загружает конфигурацию из переменных окружения.
    
    Returns:
        Словарь с настройками проекта
        
    Raises:
        ValueError: Если отсутствуют обязательные переменные
    """
    # Загружаем переменные из .env файла
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "llm_model": os.getenv("LLM_MODEL", "anthropic/claude-3-sonnet"),
        "bot_name": os.getenv("BOT_NAME", "Cooking Assistant"),
    }
    
    # Проверка обязательных переменных
    missing_vars = []
    if not config["bot_token"]:
        missing_vars.append("TELEGRAM_BOT_TOKEN")
    if not config["api_key"]:
        missing_vars.append("OPENROUTER_API_KEY")
    
    if missing_vars:
        raise ValueError(f"Отсутствуют обязательные переменные окружения: {', '.join(missing_vars)}")
    
    return config