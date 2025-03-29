from datetime import datetime

from siyuan_ai_companion.model import RagDriver, SiyuanApi


async def update_index():
    """
    Update the vector index with new blocks
    """
    try:
        with open('last_update', 'r') as f:
            last_update = int(f.read())
    except FileNotFoundError:
        last_update = 0

    async with SiyuanApi() as siyuan:
        last_update_datetime = datetime.fromtimestamp(last_update)
        current_time = datetime.now()

        blocks = await siyuan.get_blocks_by_time(
            updated_after=last_update_datetime,
        )

        updated_content: list[tuple[str, str]] = []

        for block in blocks:
            block_id = block['id']
            block_content = block['content']

            updated_content.append((block_id, block_content))

        if updated_content:
            rag_driver = RagDriver()

            rag_driver.add_blocks(
                blocks=updated_content,
            )

            with open('last_update', 'w') as f:
                f.write(str(int(current_time.timestamp())))
