"""Клиент для работы с Openrouter API."""
import logging
from openai import OpenAI


logger = logging.getLogger(__name__)


class LLMClient:
    """Клиент для взаимодействия с LLM через Openrouter."""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-sonnet", base_url: str = "https://openrouter.ai/api/v1"):
        """Инициализация клиента.
        
        Args:
            api_key: API ключ Openrouter
            model: Модель LLM для использования
            base_url: Базовый URL API
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
    
    async def get_response(self, user_message: str, system_prompt: str = None) -> str:
        """Получает ответ от LLM на основе сообщения пользователя.
        
        Args:
            user_message: Сообщение от пользователя
            system_prompt: Системный промпт (опционально)
            
        Returns:
            Ответ от LLM
            
        Raises:
            Exception: При ошибках API
        """
        try:
            # Детализированный системный промпт кулинарного помощника
            if system_prompt is None:
                system_prompt = (
                    "Ты - профессиональный кулинарный помощник с глубокими знаниями в кулинарии. "
                    "Твоя основная задача - помогать пользователям с любыми вопросами о готовке.\n\n"
                    "Основные направления помощи:\n"
                    "• Рецепты: предоставляй подробные рецепты с ингредиентами и пошаговыми инструкциями\n"
                    "• Замена ингредиентов: предлагай альтернативы при отсутствии нужных продуктов\n"
                    "• Советы по оборудованию: рекомендации по кухонной технике и утвари\n"
                    "• Техники приготовления: объясняй различные методы готовки\n"
                    "• Хранение продуктов: советы по сохранению свежести\n"
                    "• Диетические рекомендации: помощь при ограничениях в питании\n\n"
                    "Формат ответов:\n"
                    "• Будь конкретным и практичным\n"
                    "• Используй понятный язык без излишней сложности\n"
                    "• Предоставляй точные пропорции и время приготовления\n"
                    "• Учитывай доступность ингредиентов в России\n"
                    "• Будь дружелюбным и терпеливым\n\n"
                    "Если вопрос не связан с кулинарией, вежливо сообщи, что специализируешься только на кулинарных темах "
                    "и предложи обратиться к другим источникам по соответствующей теме."
                )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Ошибка при запросе к LLM: {e}")
            raise