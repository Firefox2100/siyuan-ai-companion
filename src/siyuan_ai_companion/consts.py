import os
import logging


SIYUAN_URL = os.getenv(
    'SIYUAN_URL',
    'http://localhost:6806'
)
SIYUAN_TOKEN = os.getenv(
    'SIYUAN_TOKEN',
    None
)
QDRANT_LOCATION = os.getenv(
    'QDRANT_LOCATION',
    'http://localhost:6333'
)
QDRANT_COLLECTION_NAME = os.getenv(
    'QDRANT_COLLECTION_NAME',
    'siyuan_ai_companion'
)
OPENAI_URL = os.getenv(
    'OPENAI_URL',
    'https://api.openai.com/v1/'
)

LOGGER = logging.getLogger('siyuan-ai-companion')
LOGGER.setLevel(logging.INFO)

FORCE_UPDATE_INDEX = os.getenv(
    'FORCE_UPDATE_INDEX',
    'false'
).lower() == 'true'
