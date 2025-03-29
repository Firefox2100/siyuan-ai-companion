import os


SIYUAN_URL = os.getenv(
    'SIYUAN_URL',
    'http://localhost:6806'
)
SIYUAN_TOKEN = os.getenv(
    'SIYUAN_TOKEN',
    None
)
QDRANT_COLLECTION_NAME = os.getenv(
    'QDRANT_COLLECTION_NAME',
    'siyuan_ai_companion'
)
OPENAI_URL = os.getenv(
    'OPENAI_URL',
    'https://api.openai.com/v1/'
)
