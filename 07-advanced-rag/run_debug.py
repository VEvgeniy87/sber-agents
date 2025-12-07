import os
import logging
import sys

# Устанавливаем уровень логирования DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Устанавливаем переменную окружения для оффлайн режима
os.environ['HF_HUB_OFFLINE'] = '1'

# Запускаем бота
from src.bot import main

if __name__ == '__main__':
    main()