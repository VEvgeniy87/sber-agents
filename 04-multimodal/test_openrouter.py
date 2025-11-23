#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы с OpenRouter
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llm import get_transaction_response_text

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openrouter():
    """Тестирует работу с OpenRouter API"""
    print("=== Тестирование интеграции с OpenRouter ===")
    
    test_messages = [
        "Потратил 500 рублей на продукты в Пятерочке",
        "Получил зарплату 50000 рублей",
        "Купил билеты в кино на 1200 рублей"
    ]
    
    for message in test_messages:
        print(f"\n--- Тестовое сообщение: '{message}' ---")
        
        try:
            response = await get_transaction_response_text(message, [])
            
            if response.transactions:
                print(f"[OK] Найдено транзакций: {len(response.transactions)}")
                for i, transaction in enumerate(response.transactions, 1):
                    print(f"   {i}. {transaction.type.value}: {transaction.amount} руб. ({transaction.category})")
            else:
                print("[INFO] Транзакции не найдены")
                
            print(f"[RESPONSE] Ответ LLM: {response.answer}")
            
        except Exception as e:
            print(f"[ERROR] Ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_openrouter())