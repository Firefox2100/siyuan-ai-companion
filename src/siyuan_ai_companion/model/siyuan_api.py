from collections import defaultdict
from datetime import datetime
from httpx import AsyncClient

from siyuan_ai_companion.consts import SIYUAN_URL, SIYUAN_TOKEN
from siyuan_ai_companion.errors import SiYuanApiError


class SiyuanApi:
    """
    SiYuan API client
    """
    def __init__(self,
                 url: str = None,
                 token: str = None,
                 client: AsyncClient = None,
                 ):
        """
        SiYuan API client

        :param url: The URL of the SiYuan server, for example
                    http://localhost:6806, leave empty to use
                    the default from the environment variable
        :param token: The token for the SiYuan server, leave
                      empty to use the default from the environment
        :param client: The httpx client to use, leave empty to
                       create a new one
        """
        self.url = url or SIYUAN_URL
        self.token = token or SIYUAN_TOKEN

        headers = None
        if token is not None:
            headers = {
                'Authorization': f'Token {token}'
            }

        self._client = client if client is not None else AsyncClient(
            base_url=self.url,
            headers=headers,
        )

        self._block_count: int | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """
        Close the httpx client

        Be careful if the client is passed in from outside,
        as this will close the client and it will not be
        usable any more.
        """
        await self._client.aclose()

    async def _raw_query(self,
                         sql_query: str,
                         ) -> list[dict]:
        """
        Execute a raw SQL query on the SiYuan database

        :param sql_query: The SQL query to execute
        :return: The data response from the SiYuan server
        """
        response = await self._client.post(
            url='/api/query/sql',
            json={
                'stmt': sql_query,
            }
        )

        if response.status_code != 200:
            raise SiYuanApiError(
                message='Failed to execute SQL query',
                status_code=response.status_code,
            )
        if response.json().get('code') != 0:
            raise SiYuanApiError(
                message=response.json()['msg'],
                status_code=response.status_code,
            )

        data = response.json()
        return data['data']

    @staticmethod
    def _sort_nodes(nodes: list[dict]) -> list[dict]:
        """
        Sorting function for SiYuan blocks within the same note

        Sorting rules:
        - Nodes with no parent ID are considered root nodes
        - Nodes are sorted by their sort value, ascending order
        - Parent comes before children
        - Children stays with its parent, before other nodes in the
          parent level

        :param nodes: The list of nodes to sort
        :return: The sorted list of nodes
        """
        children_map = defaultdict(list)
        roots = []

        for obj in nodes:
            if obj['parent_id']:
                children_map[obj['parent_id']].append(obj)
            else:
                roots.append(obj)

        def walk(node_list):
            node_list.sort(key=lambda x: x['sort'])
            result = []
            for node in node_list:
                result.append(node)
                children = children_map.get(node['id'], [])
                result.extend(walk(children))
            return result

        return walk(roots)

    async def get_count(self) -> int:
        """
        Get the number of blocks in the database
        """
        payload = await self._raw_query(
            sql_query="SELECT COUNT(*) FROM blocks",
        )

        self._block_count = int(payload[0]['COUNT(*)'])
        return self._block_count

    async def get_block(self,
                        block_id: str,
                        ) -> dict | None:
        """
        Get a block by its ID

        :param block_id: The ID of the block, used in SiYuan
        """
        payload = await self._raw_query(
            sql_query=f"SELECT * FROM blocks WHERE id='{block_id}'",
        )

        return payload[0] if payload else None

    async def get_blocks_by_time(self,
                                 updated_after: datetime = None,
                                 ) -> list[dict]:
        """
        Get all blocks updated after a certain time

        :param updated_after: The time to filter by. If left empty,
                              all blocks will be returned.
        :return: A list of blocks updated after the given time
        """
        if updated_after is None:
            updated_after = datetime.fromtimestamp(0)

        updated_after_str = updated_after.strftime("%Y%m%d%H%M%S")

        if self._block_count is None:
            await self.get_count()

        payload = await self._raw_query(
            sql_query=f"SELECT * FROM blocks WHERE updated > '{updated_after_str}'"
                      f"LIMIT {self._block_count}",
        )

        return payload

    async def get_blocks_by_note(self,
                                 note_id: str,
                                 ) -> list[dict]:
        """
        Get all blocks in a note by its ID

        :param note_id: The ID of the note, used in SiYuan
        :return: A list of blocks in the note
        """
        if self._block_count is None:
            await self.get_count()

        payload = await self._raw_query(
            f"SELECT * FROM blocks WHERE root_id = '{note_id}' LIMIT {self._block_count}"
        )

        payload = self._sort_nodes(payload)

        return payload

    async def get_note_plaintext(self,
                                 note_id: str,
                                 ) -> str:
        """
        Get the plaintext of a note by its ID

        :param note_id: The ID of the note, used in SiYuan
        :return: The plaintext of the note
        """
        blocks = await self.get_blocks_by_note(
            note_id=note_id,
        )

        contents = [
            block.get('content', '') for block in blocks
        ]

        content = '\n'.join(contents)

        # Remove the ZWSP and multiple consecutive newlines
        content = content.replace('\u200b', '')
        content_lines = content.split('\n')
        content_lines = [line.strip() for line in content_lines if line.strip()]

        # Remove duplicated lines
        content_lines = list(dict.fromkeys(content_lines))
        content = '\n'.join(content_lines)

        return content
