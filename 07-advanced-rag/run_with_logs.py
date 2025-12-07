import os
import sys
import logging

# Устанавливаем уровень логирования DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

os.environ['HF_HUB_OFFLINE'] = '1'

# Импортируем и запускаем бота
from src.bot import main

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception("Bot crashed")
        sys.exit(1)