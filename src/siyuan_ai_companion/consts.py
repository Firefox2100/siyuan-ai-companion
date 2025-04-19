"""
Constant values and configuration for the Siyuan AI Companion project.
"""

import os
import logging
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
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
        data_dir = user_data_dir(APP_CONFIG.application_name)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    return data_dir


class AppConfig(BaseSettings):
    """
    Application configuration settings.
    """
    # Core service configuration
    application_name: str = 'siyuan-ai-companion'

    # SiYuan
    siyuan_url: str = Field(
        'http://localhost:6806',
        description='SiYuan API URL root'
    )
    siyuan_token: Optional[str] = Field(
        None,
        description='SiYuan API token (not auth code)'
    )

    # Qdrant
    qdrant_location: str = Field(
        'http://localhost:6333',
        description='Qdrant database location'
    )
    qdrant_collection_name: str = Field(
        'siyuan_ai_companion',
        description='Qdrant collection name'
    )

    # OpenAI
    openai_url: str = Field(
        'https://api.openai.com/v1/',
        description='OpenAI-compatible proxy URL'
    )
    openai_token: Optional[str] = Field(
        None,
        description='OpenAI API token'
    )

    # Service Authentication
    companion_token: Optional[str] = Field(
        None,
        description='Token to authenticate with this service'
    )

    # Runtime Behaviour
    whisper_workers: int = Field(
        1,
        description='Number of workers for faster-whisper'
    )
    huggingface_hub_token: Optional[str] = Field(
        None,
        description='HF Hub token for model downloads'
    )
    siyuan_transcribe_notebook: Optional[str] = Field(
        None,
        description='Notebook to store transcribed notes'
    )

    # Debug / Override flags
    force_update_index: bool = Field(
        False,
        description='Force update index on startup'
    )


APP_CONFIG = AppConfig()

# Utility
LOGGER = logging.getLogger('siyuan-ai-companion')
LOGGER.setLevel(logging.INFO)
DATA_DIR = _get_data_dir()
