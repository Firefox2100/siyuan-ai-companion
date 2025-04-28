"""
Constant values and configuration for the Siyuan AI Companion project.
"""

import logging
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


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
    companion_logging_level: str = Field(
        'INFO',
        description='Logging level for the companion service'
    )


APP_CONFIG = AppConfig()

# Utility
LOGGER = logging.getLogger('siyuan-ai-companion')
LOGGER.setLevel(APP_CONFIG.companion_logging_level)

if not LOGGER.hasHandlers():
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(APP_CONFIG.companion_logging_level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    LOGGER.addHandler(console_handler)


LOGGER.debug('Application configuration: %s', APP_CONFIG.model_dump_json(indent=2))
