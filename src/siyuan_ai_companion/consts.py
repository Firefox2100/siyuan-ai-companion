"""
Constant values and configuration for the Siyuan AI Companion project.
"""

import os
import logging
from platformdirs import user_data_dir


def _get_data_dir():
    """
    Get the data directory for the application.
    :return: The data directory as a string.
    """
    data_dir = os.getenv(
        'SIYUAN_AI_COMPANION_DATA_DIR',
    )

    if not data_dir:
        # Use default data directory
        data_dir = user_data_dir(APPLICATION_NAME)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    return data_dir


# Configuration constants
APPLICATION_NAME = 'siyuan-ai-companion'
SIYUAN_URL = os.getenv(                 # SiYuan API URL root
    'SIYUAN_URL',
    'http://localhost:6806'
)
SIYUAN_TOKEN = os.getenv(               # SiYuan API (NOT AUTHORISATION CODE) TOKEN
    'SIYUAN_TOKEN',
    None
)
QDRANT_LOCATION = os.getenv(            # Qdrant database location, can be :memory:
    'QDRANT_LOCATION',
    'http://localhost:6333'
)
QDRANT_COLLECTION_NAME = os.getenv(     # Qdrant collection name
    'QDRANT_COLLECTION_NAME',
    'siyuan_ai_companion'
)
OPENAI_URL = os.getenv(                 # OpenAI compatible URL to proxy to
    'OPENAI_URL',
    'https://api.openai.com/v1/'
)
OPENAI_TOKEN = os.getenv(               # OpenAI API token
    'OPENAI_TOKEN',
    None
)
COMPANION_TOKEN = os.getenv(            # Token to authenticate with this service
    'COMPANION_TOKEN',
    None
)
WHISPER_WORKERS = int(os.getenv(        # Number of workers to use for faster-whisper
    'WHISPER_WORKERS',
    '1'
))
HUGGINGFACE_HUB_TOKEN = os.getenv(      # Hugging Face Hub token for downloading models
    'HUGGINGFACE_HUB_TOKEN',
    None
)


# Overridable configurations for runtime behavior
SIYUAN_TRANSCRIBE_NOTEBOOK = os.getenv( # Where to store the transcribed notes
    'SIYUAN_TRANSCRIBE_NOTEBOOK',
    None
)


# Debug or behavior overrides
FORCE_UPDATE_INDEX = os.getenv(         # Force update the index on startup
    'FORCE_UPDATE_INDEX',
    'false'
).lower() == 'true'


# Utility
LOGGER = logging.getLogger('siyuan-ai-companion')
LOGGER.setLevel(logging.INFO)
DATA_DIR = _get_data_dir()
