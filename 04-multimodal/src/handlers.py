import logging
import base64
import asyncio
from datetime import time
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from openai import APIError, InternalServerError
from llm import get_transaction_response_text, get_transaction_response_image
from transcribe_alt import transcribe_voice_message_alt
from transcribe_vosk import transcribe_voice_message_vosk
from transcribe import transcriber
from models import Transaction
from config import config

logger = logging.getLogger(__name__)
router = Router()

# Глобальные словари для хранения данных
chat_conversations: dict[int, list[dict]] = {}
transactions: dict[int, list[Transaction]] = {}

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000

@router.message(Command("start"))
async def cmd_start(message: Message):
    chat_id = message.chat.id
    logger.info(f"User {chat_id} started the bot")
    
    # Очищаем историю и транзакции для данного чата
    chat_conversations[chat_id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
    ]
    transactions[chat_id] = []
    
    await message.answer(
        "Привет! Я персональный финансовый советник.\n\n"
        "Я могу:\n"
        "• Извлекать транзакции из ваших сообщений\n"
        "• Распознавать транзакции из голосовых сообщений\n"
        "• Вести учет доходов и расходов\n"
        "• Предоставлять советы по управлению финансами\n\n"
        "Используйте /start для начала нового диалога и очистки истории."
    )

@router.message(Command("balance"))
async def cmd_balance(message: Message):
    chat_id = message.chat.id
    logger.info(f"Balance requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "💵 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Расчет баланса, доходов и расходов
    total_income = sum(t.amount for t in user_transactions if t.type.value == "income")
    total_expense = sum(t.amount for t in user_transactions if t.type.value == "expense")
    balance = total_income - total_expense
    
    # Статистика по категориям
    category_stats: dict[str, float] = {}
    for t in user_transactions:
        category = t.category
        if category not in category_stats:
            category_stats[category] = 0.0
        if t.type.value == "income":
            category_stats[category] += t.amount
        else:
            category_stats[category] -= t.amount
    
    # Форматирование отчета
    report_lines = [
        "💵 **Отчет о балансе**\n",
        f"📊 Баланс: {balance:.2f} руб.",
        f"💰 Доходы: {total_income:.2f} руб.",
        f"💸 Расходы: {total_expense:.2f} руб.",
        f"\n📈 Всего транзакций: {len(user_transactions)}",
        "\n**Статистика по категориям:**"
    ]
    
    # Сортируем категории по сумме (от большей к меньшей)
    sorted_categories = sorted(category_stats.items(), key=lambda x: abs(x[1]), reverse=True)
    for category, amount in sorted_categories:
        sign = "💰" if amount > 0 else "💸"
        report_lines.append(f"{sign} {category}: {amount:+.2f} руб.")
    
    await message.answer("\n".join(report_lines))

@router.message(Command("transactions"))
async def cmd_transactions(message: Message):
    chat_id = message.chat.id
    logger.info(f"Transactions list requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "📋 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Сортируем транзакции по дате (от новых к старым)
    sorted_transactions = sorted(user_transactions, key=lambda t: (t.date, t.time or time(0, 0)), reverse=True)
    
    # Форматирование списка транзакций
    report_lines = [
        f"📋 **Все транзакции** ({len(user_transactions)} шт.)\n"
    ]
    
    for i, t in enumerate(sorted_transactions, 1):
        # Форматирование даты и времени
        date_str = t.date.strftime("%d.%m.%Y")
        time_str = f" {t.time.strftime('%H:%M')}" if t.time else ""
        
        # Знак и тип транзакции
        sign = "💰" if t.type.value == "income" else "💸"
        type_str = "Доход" if t.type.value == "income" else "Расход"
        
        # Форматирование суммы
        amount_str = f"{t.amount:.2f}".rstrip('0').rstrip('.')
        
        # Описание (если есть)
        desc_str = f"\n   {t.description}" if t.description else ""
        
        report_lines.append(
            f"{i}. {sign} **{type_str}** {amount_str} руб.\n"
            f"   📅 {date_str}{time_str}\n"
            f"   🏷️ {t.category}{desc_str}"
        )
    
    # Если транзакций много, разбиваем на несколько сообщений (Telegram лимит ~4096 символов)
    report_text = "\n\n".join(report_lines)
    if len(report_text) > 4000:
        # Разбиваем на части
        parts = []
        current_part = [report_lines[0]]  # Заголовок
        current_length = len(report_lines[0])
        
        for line in report_lines[1:]:
            line_length = len(line) + 2  # +2 для "\n\n"
            if current_length + line_length > 4000:
                parts.append("\n\n".join(current_part))
                current_part = [line]
                current_length = len(line)
            else:
                current_part.append(line)
                current_length += line_length
        
        if current_part:
            parts.append("\n\n".join(current_part))
        
        # Отправляем части
        for part in parts:
            await message.answer(part)
    else:
        await message.answer(report_text)

@router.message(lambda message: message.photo or (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")))
async def handle_image(message: Message):
    chat_id = message.chat.id
    
    logger.info(f"Image received from {chat_id}")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE}
        ]
    
    # Получаем историю сообщений без системного промпта для контекста
    message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
    
    try:
        # Получаем изображение в формате base64
        if message.photo:
            # Берем самое большое фото
            photo = message.photo[-1]
            file_info = await message.bot.get_file(photo.file_id)
            file_bytes = await message.bot.download_file(file_info.file_path)
        elif message.document:
            file_info = await message.bot.get_file(message.document.file_id)
            file_bytes = await message.bot.download_file(file_info.file_path)
        else:
            await message.answer("Не удалось получить изображение.")
            return
        
        # Конвертируем в base64
        import base64
        image_base64 = base64.b64encode(file_bytes.read()).decode('utf-8')
        
        # Получаем ответ LLM для изображения
        response = await get_transaction_response_image(image_base64, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM image response for {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from image for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from image for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''} из изображения"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены в изображении"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error for image {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        if "402" in error_message or "credits" in error_message.lower():
            await message.answer(
                "📷 Извините, для обработки изображений требуется пополнить баланс на OpenRouter.\n\n"
                "В настоящее время бесплатные модели для распознавания изображений недоступны.\n\n"
                "Вы можете:\n"
                "• Отправить текстовое описание транзакции (работает отлично!)\n"
                "• Использовать команды /balance и /transactions\n"
                "• Начать новый диалог с помощью /start\n\n"
                "Пример текстовой транзакции: \"Потратил 500 рублей на продукты в Пятерочке\""
            )
        elif "404" in error_message or "not found" in error_message.lower():
            await message.answer(
                "📷 Извините, модель для обработки изображений временно недоступна.\n\n"
                "В настоящее время модели для распознавания чеков и скриншотов не работают на OpenRouter.\n\n"
                "Вы можете:\n"
                "• Отправить текстовое описание транзакции\n"
                "• Использовать команды /balance и /transactions\n"
                "• Начать новый диалог с помощью /start\n\n"
                "Текстовые транзакции работают отлично! Например: \"Получил зарплату 50000 рублей\""
            )
        else:
            await message.answer(
                "Извините, произошла ошибка на стороне провайдера LLM при обработке изображения. "
                "Пожалуйста, попробуйте еще раз через несколько секунд."
            )
    except Exception as e:
        logger.error(f"Error in handle_image for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке вашего изображения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

@router.message(lambda message: message.voice or message.audio)
async def handle_voice(message: Message):
    chat_id = message.chat.id
    
    logger.info(f"Voice/audio message received from {chat_id}")
    
    try:
        # Скачиваем голосовое сообщение
        if message.voice:
            file_info = await message.bot.get_file(message.voice.file_id)
        elif message.audio:
            file_info = await message.bot.get_file(message.audio.file_id)
        else:
            await message.answer("Не удалось получить голосовое сообщение.")
            return
        
        file_bytes = await message.bot.download_file(file_info.file_path)
        
        # Пробуем использовать локальный Vosk сначала
        transcribed_text = await transcribe_voice_message_vosk(file_bytes)
        
        # Если Vosk не сработал, пробуем Whisper
        if not transcribed_text:
            logger.warning("Vosk transcription failed, trying Whisper")
            try:
                # Создаем временный файл для Whisper
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
                    temp_file.write(file_bytes.read())
                    temp_file_path = temp_file.name
                
                # Транскрибируем с помощью Whisper
                transcribed_text = await transcriber.transcribe_voice(temp_file_path)
                
                # Очищаем временный файл
                transcriber.cleanup_temp_file(temp_file_path)
                
                if transcribed_text:
                    logger.info(f"Whisper transcription successful: '{transcribed_text}'")
                else:
                    logger.warning("Whisper returned None, using alternative method")
                    
            except Exception as e:
                logger.warning(f"Whisper transcription failed, using alternative: {e}")
        
        # Если ни Vosk ни Whisper не сработали, используем альтернативный метод
        if not transcribed_text:
            file_bytes.seek(0)  # Сбрасываем позицию файла
            transcribed_text = await transcribe_voice_message_alt(file_bytes)
        
        if not transcribed_text:
            await message.answer(
                "Не удалось распознать речь в голосовом сообщении. "
                "Пожалуйста, попробуйте еще раз или отправьте текстовое сообщение."
            )
            return
        
        logger.info(f"Voice transcription for {chat_id}: '{transcribed_text}'")
        
        # Инициализируем историю если её нет
        if chat_id not in chat_conversations:
            chat_conversations[chat_id] = [
                {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
            ]
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        # Получаем ответ LLM с structured output
        response = await get_transaction_response_text(transcribed_text, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for voice {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from voice for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from voice message for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''} из голосового сообщения"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем распознанный текст в историю как сообщение пользователя
        chat_conversations[chat_id].append(
            {"role": "user", "content": transcribed_text}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        # Пытаемся отправить сообщение с повторными попытками
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await message.answer(answer_text)
                break  # Успешно отправлено
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Ждем перед повторной попыткой
                else:
                    raise  # Все попытки исчерпаны
        
    except Exception as e:
        logger.error(f"Error in handle_voice for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке голосового сообщения. "
            "Попробуйте еще раз или отправьте текстовое сообщение."
        )

@router.message()
async def handle_message(message: Message):
    chat_id = message.chat.id
    
    # Пропускаем сообщения без текста (обрабатываются другими обработчиками)
    if not message.text or message.text.strip() == "":
        return
    
    logger.info(f"Text message received from {chat_id}: '{message.text[:100]}...'")
    
    try:
        # Инициализируем историю если её нет
        if chat_id not in chat_conversations:
            chat_conversations[chat_id] = [
                {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
            ]
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        # Получаем ответ LLM с structured output
        response = await get_transaction_response_text(message.text, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for text {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from text for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from text message for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем сообщение пользователя в историю
        chat_conversations[chat_id].append(
            {"role": "user", "content": message.text}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
        
    except Exception as e:
        logger.error(f"Error in handle_message for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )
