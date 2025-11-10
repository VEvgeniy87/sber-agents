"""Тестовый скрипт для проверки работы LLM клиента."""
import asyncio
import sys
import os

# Добавляем путь к src для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm_client import LLMClient
from src.config import load_config


async def test_llm_client():
    """Тестирует работу LLM клиента с реальным API."""
    try:
        # Загружаем конфигурацию
        config = load_config()
        print("Конфигурация загружена успешно")
        
        # Создаем клиент LLM
        client = LLMClient(
            api_key=config["api_key"],
            model=config["llm_model"]
        )
        print(f"LLM клиент создан (модель: {config['llm_model']})")
        
        # Тестовый кулинарный вопрос
        test_question = "Как приготовить пасту карбонару? Опиши простой рецепт."
        print(f"\nТестовый вопрос: {test_question}")
        
        # Получаем ответ от LLM
        print("Отправляем запрос к LLM...")
        response = await client.get_response(test_question)
        
        print(f"\nLLM ответил успешно!")
        print(f"Ответ ({len(response)} символов):")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return True
        
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}")
        return False
    except Exception as e:
        print(f"Ошибка при работе с LLM: {e}")
        return False


if __name__ == "__main__":
    print("Тестирование LLM клиента с Openrouter API")
    print("=" * 50)
    
    success = asyncio.run(test_llm_client())
    
    if success:
        print("\nТест пройден успешно! LLM клиент работает корректно.")
    else:
        print("\nТест не пройден. Проверьте конфигурацию и API ключ.")
        sys.exit(1)