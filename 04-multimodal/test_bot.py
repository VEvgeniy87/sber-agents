import asyncio
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from llm import get_transaction_response_text
from models import TransactionResponse

async def test_bot_connection():
    print("=== Testing bot connection to Ollama server ===")
    print(f"URL: {config.OPENAI_BASE_URL}")
    print(f"Text model: {config.MODEL_TEXT}")
    print(f"Image model: {config.MODEL_IMAGE}")
    
    # Test text model
    print("\n--- Testing text model ---")
    try:
        response = await get_transaction_response_text(
            "Hello! Reply briefly.",
            []
        )
        print(f"SUCCESS: Text model works")
        print(f"Answer: {response.answer}")
        print(f"Transactions: {len(response.transactions)}")
    except Exception as e:
        print(f"ERROR: Text model error: {e}")
    
    # Test transaction extraction
    print("\n--- Testing transaction extraction ---")
    try:
        test_message = "Bought coffee for 300 rubles at Starbucks"
        response = await get_transaction_response_text(
            test_message,
            []
        )
        print(f"SUCCESS: Transaction extraction works")
        print(f"Answer: {response.answer}")
        print(f"Transactions: {len(response.transactions)}")
        if response.transactions:
            for tx in response.transactions:
                print(f"  - {tx.amount} rub. - {tx.description}")
    except Exception as e:
        print(f"ERROR: Transaction extraction error: {e}")

if __name__ == "__main__":
    asyncio.run(test_bot_connection())