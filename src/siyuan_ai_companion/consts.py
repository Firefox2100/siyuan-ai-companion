import os


SIYUAN_DATA_DIR = os.getenv(
    'SIYUAN_DATA_DIR',
    os.path.expanduser('~/.var/app/org.b3log.siyuan/SiYuan/data')
)
QDRANT_COLLECTION_NAME = os.getenv(
    'QDRANT_COLLECTION_NAME',
    'siyuan_ai_companion'
)
OPENAI_URL = os.getenv(
    'OPENAI_URL',
    'https://api.openai.com/v1/'
)
