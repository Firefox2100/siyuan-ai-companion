"""
This module contains the task functions for running in
the background
"""

from datetime import datetime

from siyuan_ai_companion.consts import LOGGER
from siyuan_ai_companion.model import RagDriver, SiyuanApi


async def update_index():
    """
    Update the vector index with new blocks
    """
    LOGGER.info("Updating vector index")

    try:
        with open('last_update', encoding='utf-8') as f:
            last_update = int(f.read())
    except FileNotFoundError:
        last_update = 0

    async with SiyuanApi() as siyuan:
        last_update_datetime = datetime.fromtimestamp(last_update)
        current_time = datetime.now()

        blocks = await siyuan.get_blocks_by_time(
            updated_after=last_update_datetime,
        )

        LOGGER.info('%s blocks updated since last update', len(blocks))

        updated_content: list[tuple[str, str, str]] = []

        for block in blocks:
            block_id = block['id']
            block_content = block['content']
            document_id = block['root_id']

            updated_content.append((block_id, block_content, document_id))

        if updated_content:
            rag_driver = RagDriver()

            rag_driver.update_blocks(
                blocks=updated_content,
            )

            with open('last_update', 'w', encoding='utf-8') as f:
                f.write(str(int(current_time.timestamp())))

            LOGGER.info('Index updated successfully')
