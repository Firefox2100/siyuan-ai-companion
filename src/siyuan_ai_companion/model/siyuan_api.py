"""
SiYuan API client

It handles the communication with the SiYuan server, querying
blocks, tracking updates and retrieving notes.
"""

import re
from copy import deepcopy
from contextlib import asynccontextmanager
from typing import AsyncIterator
from asyncio import Lock
from tempfile import NamedTemporaryFile
from datetime import datetime
from httpx import AsyncClient, Response

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from siyuan_ai_companion.errors import SiYuanApiError, SiYuanFileListError, \
    SiYuanBlockNotFoundError


class SiyuanApi:
    """
    SiYuan API client
    """
    _processing_assets: set[str] = set()
    _asset_lock = Lock()

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
        self.url = url or APP_CONFIG.siyuan_url
        self.token = token or APP_CONFIG.siyuan_token

        headers = None
        if self.token is not None:
            LOGGER.info('Authentication for SiYuan API enabled')
            LOGGER.debug('Token: %s', self.token)

            headers = {
                'Authorization': f'Token {self.token}'
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
        LOGGER.debug('Closing SiYuan API client')

        await self._client.aclose()

    @classmethod
    async def add_to_processing(cls,
                                asset_path: str,
                                ):
        """
        Add an asset to the processing list
        :param asset_path: The path of the asset being processed
        """
        async with cls._asset_lock:
            if asset_path in cls._processing_assets:
                LOGGER.debug('Asset %s is already being processed', asset_path)
                return

            cls._processing_assets.add(asset_path)
            LOGGER.debug('Added asset %s to processing list', asset_path)

    @classmethod
    async def remove_from_processing(cls,
                                     asset_path: str,
                                     ):
        """
        Remove an asset from the processing list
        :param asset_path: The path of the asset being processed
        """
        async with cls._asset_lock:
            if asset_path not in cls._processing_assets:
                LOGGER.debug('Asset %s is not in processing list', asset_path)
                return

            cls._processing_assets.remove(asset_path)
            LOGGER.debug('Removed asset %s from processing list', asset_path)

    @classmethod
    async def is_processing(cls,
                            asset_path: str,
                            ) -> bool:
        """
        Check if an asset is being processed
        :param asset_path: The path of the asset being processed
        :return: True if the asset is being processed, False otherwise
        """
        async with cls._asset_lock:
            is_processing = asset_path in cls._processing_assets
            LOGGER.debug('Asset %s is being processed: %s', asset_path, is_processing)
            return is_processing

    async def _raw_post(self,
                        url: str,
                        payload: dict | None = None,
                        response_is_json: bool = True,
                        ) -> dict | list | str | Response:
        """
        Execute a raw POST request on the SiYuan server
        :param url: The URL to post to
        :param payload: The JSON payload to send
        :param response_is_json: If the response is JSON format
        :return: A dictionary if the response is JSON format,
                 or the raw response
        :raises SiYuanApiError: If the request fails
        """
        LOGGER.debug('POST %s with payload %s', url, payload)

        response = await self._client.post(
            url=url,
            json=payload,
        )

        if response.status_code != 200:
            LOGGER.error(
                'POST %s with payload %s failed with status code %d',
                url,
                payload,
                response.status_code,
            )

            raise SiYuanApiError(
                message='Failed to communicate with SiYuan server',
                status_code=response.status_code,
            )

        if response_is_json:
            if response.json().get('code') != 0:
                LOGGER.error(
                    'POST %s with payload %s failed with error: %s',
                    url,
                    payload,
                    response.json()['msg'],
                )

                raise SiYuanApiError(
                    message=response.json()['msg'],
                    status_code=response.status_code,
                )

            data = response.json()

            LOGGER.debug(
                'POST %s with payload %s returned data: %s',
                url,
                payload,
                data,
            )

            return data['data']

        # If the response is not JSON, return the raw response
        LOGGER.debug(
            'POST %s with payload %s returned raw response',
            url,
            payload,
        )

        return response

    async def _raw_query(self,
                         sql_query: str,
                         ) -> list[dict]:
        """
        Execute a raw SQL query on the SiYuan database

        :param sql_query: The SQL query to execute
        :return: The data response from the SiYuan server
        :raises SiYuanApiError: If the request fails
        """
        return await self._raw_post(
            url='/api/query/sql',
            payload={
                'stmt': sql_query,
            },
        )

    async def _list_files_recursive(self,
                                    path: str,
                                    ) -> list[str]:
        """
        List all assets in a given path recursively
        :param path: The path to begin the search
        :return: All files under the path, including children dirs
        :raises SiYuanFileListError: If the request format is unexpected.
            Usually this means the path is invalid, or there's a racing
            condition with the SiYuan server.
        :raises SiYuanApiError: If the request fails
        """
        if path != '/':
            path = path.rstrip('/')

        response = await self._raw_post(
            url='/api/file/readDir',
            payload={
                'path': path,
            }
        )

        files = []

        for item in response:
            if not isinstance(item, dict):
                LOGGER.error(
                    'Unexpected response type from API: %s',
                    type(item),
                )

                raise SiYuanFileListError(
                    message='Unexpected response type from API',
                )

            item_path = f'{path}/{item["name"]}'

            if item['isDir']:
                # It's a directory, list its contents
                children = await self._list_files_recursive(
                    path=item_path,
                )
                files.extend(children)
            else:
                # It's a file, add it to the list
                files.append(item_path)

        LOGGER.info('Listed %d files in %s', len(files), path)
        LOGGER.debug('Listed files in %s: %s', path, files)

        return files

    async def get_count(self) -> int:
        """
        Get the number of blocks in the database
        :return: The number of blocks in the database
        :raises SiYuanApiError: If the request fails
        """
        payload = await self._raw_query(
            sql_query="SELECT COUNT(*) FROM blocks",
        )

        self._block_count = int(payload[0]['COUNT(*)'])

        LOGGER.info('Counted %d blocks', self._block_count)

        return self._block_count

    async def get_block(self,
                        block_id: str,
                        ) -> dict | None:
        """
        Get a block by its ID

        :param block_id: The ID of the block, used in SiYuan
        :return: The block data
        :raises SiYuanApiError: If the request fails
        """
        payload = await self._raw_query(
            sql_query=f"SELECT * FROM blocks WHERE id='{block_id}'",
        )

        if not payload:
            LOGGER.warn('Block not found: %s', block_id)
            return None

        LOGGER.info('Block found for ID %s', block_id)
        LOGGER.debug('Block found: %s', payload[0])

        return payload[0]

    async def get_audio_block(self,
                              audio_name: str,
                              ) -> str:
        """
        Get an audio block by the file name of the audio asset

        :param audio_name: The file name of the audio asset, as it is
                           on the file system
        :return: The audio block ID
        :raises SiYuanBlockNotFoundError: If the block is not found
        :raises SiYuanApiError: If the request fails
        """
        payload = await self._raw_query(
            sql_query=f"SELECT * FROM blocks "
                      f"WHERE type = 'audio' AND content LIKE '%{audio_name}%'"
        )

        if not payload:
            raise SiYuanBlockNotFoundError(
                message='Audio block not found',
                status_code=404,
            )

        LOGGER.info('Audio block found for name %s', audio_name)
        LOGGER.debug('Audio block found: %s', payload[0])

        # Find the first occasion in case the audio is inserted more than once
        return payload[0]['id']

    async def get_audio_blocks(self,
                               audio_names: list[str],
                               ) -> dict[str, str]:
        """
        Get audio blocks by the file names of the audio assets

        If the audio asset is not inserted into a block, it will not appear
        in the results.
        :param audio_names: The file names of the audio assets, as they are
                            on the file system
        :return: A dictionary mapping the audio file name to the
                 audio block ID
        :raises SiYuanApiError: If the request fails
        """
        audio_names = deepcopy(audio_names)
        response = await self._raw_query(
            sql_query="SELECT * FROM blocks WHERE type = 'audio'"
        )

        audio_blocks = {}

        for block in response:
            for audio_name in audio_names:
                if audio_name in block['content']:
                    audio_blocks[audio_name] = block['id']
                    audio_names.remove(audio_name)

        LOGGER.info('Audio blocks found for names %s', audio_names)
        LOGGER.debug('Audio blocks found: %s', audio_blocks)

        return audio_blocks

    async def get_audio_transcription_id(self,
                                         audio_id: str,
                                         ) -> str:
        """
        Get the block marking the beginning of the transcription for an audio block

        :param audio_id: The audio block ID
        :return: The beginning (usually title) of the transcription block ID
        :raises SiYuanBlockNotFoundError: If the block is not found
        :raises SiYuanApiError: If the request fails
        """
        response = await self._raw_query(
            sql_query=f"SELECT * FROM blocks WHERE alias LIKE '%transcription-{audio_id}%'"
        )

        if not response:
            LOGGER.error('Transcription block not found for audio ID %s', audio_id)
            raise SiYuanBlockNotFoundError(
                message='Transcription block not found',
                status_code=404,
            )

        LOGGER.info('Transcription block found for audio ID %s', audio_id)
        LOGGER.debug('Transcription block found: %s', response[0])

        return response[0]['id']

    async def get_audio_transcription_ids(self,
                                          audio_ids: list[str],
                                          ) -> dict[str, str]:
        """
        Get the transcript block ids for multiple audio blocks

        :param audio_ids: The audio block IDs
        :return: A dictionary mapping the audio block ID to the
                 transcription block ID
        :raises SiYuanApiError: If the request fails
        """
        audio_ids = deepcopy(audio_ids)
        response = await self._raw_query(
            sql_query="SELECT * FROM blocks WHERE alias LIKE '%transcription-%'"
        )

        transcription_ids = {}

        pattern = re.compile(r'transcription-(\d{14}-\w{7})')
        for block in response:
            # Regex extracts the audio block ID (14 digits - 7 random chars)
            audio_id = pattern.search(block['alias'])

            if audio_id:
                audio_id = audio_id.group(1)
                if audio_id in audio_ids:
                    transcription_ids[audio_id] = block['id']
                    audio_ids.remove(audio_id)

        LOGGER.info('Transcription blocks found for audio IDs %s', audio_ids)
        LOGGER.debug('Transcription blocks found: %s', transcription_ids)

        # Add the processing audio IDs to the transcription IDs
        async with self._asset_lock:
            processing_paths = list(self._processing_assets)

        processing_ids = await self.get_audio_blocks(
            audio_names=processing_paths,
        )

        for audio_name, audio_id in processing_ids.items():
            if audio_id in transcription_ids:
                transcription_ids[audio_name] = 'Processing'

        return transcription_ids

    async def get_blocks_by_time(self,
                                 updated_after: datetime = None,
                                 ) -> list[dict]:
        """
        Get all blocks updated after a certain time

        :param updated_after: The time to filter by. If left empty,
                              all blocks will be returned.
        :return: A list of blocks updated after the given time
        :raises SiYuanApiError: If the request fails
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

        LOGGER.info('%s blocks updated after %s', len(payload), updated_after_str)
        LOGGER.debug('Blocks updated after %s: %s', updated_after_str, payload)

        return payload

    async def get_note_markdown(self,
                                note_id: str,
                                ) -> str:
        """
        Get the Markdown content of a note by its ID

        :param note_id: The ID of the note, used in SiYuan
        :return: The Markdown version of the note
        :raises SiYuanApiError: If the request fails
        """
        data = await self._raw_post(
            url='/api/lute/copyStdMarkdown',
            payload={
                'id': note_id,
            },
        )

        LOGGER.info('Markdown content of note %s retrieved', note_id)
        LOGGER.debug('Markdown content of note %s: %s', note_id, data)

        return data

    @asynccontextmanager
    async def download_asset(self,
                             asset_path: str,
                             ) -> AsyncIterator:
        """
        Download an asset from the SiYuan server
        and store it as a temporary file
        :param asset_path: The asset relative path to '/data/assets'
        :return: The file like object of the downloaded asset
        :raises SiYuanApiError: If the request fails
        """
        response = await self._raw_post(
            url='/api/file/getFile',
            payload={
                'path': f'/data/assets/{asset_path}',
            },
            response_is_json=False,
        )

        content_type = response.headers.get('Content-Type', '')
        suffix = "." + content_type.split("/")[-1]

        with NamedTemporaryFile(
            suffix=suffix,
        ) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

            LOGGER.info('Downloaded asset %s to %s', asset_path, temp_file.name)

            yield temp_file

    async def list_assets(self,
                          suffixes: list[str] = None,
                          ) -> list[str]:
        """
        List all assets in the SiYuan server
        :param suffixes: A list of file suffixes to filter by
                         (e.g. ['.mp3', '.wav'])
        :return: A list of asset paths
        :raises SiYuanFileListError: If the request format is unexpected.
            Usually this means the path is invalid, or there's a racing
            condition with the SiYuan server.
        :raises SiYuanApiError: If the request fails
        """
        assets = await self._list_files_recursive(
            path='/data/assets',
        )

        if suffixes is not None:
            assets = [
                asset for asset in assets
                if any(asset.endswith(suffix) for suffix in suffixes)
            ]

        # Remove the leading '/data/assets/' from the path
        assets = [asset.replace('/data/assets/', '') for asset in assets]

        LOGGER.info('Listed %d assets', len(assets))
        LOGGER.debug('Assets listed: %s', assets)

        return assets

    async def create_note(self,
                          notebook_id: str,
                          path: str,
                          markdown_content: str,
                          ) -> str:
        """
        Create a note in given path from Markdown content
        :param notebook_id: The ID of the notebook
        :param path: The `hpath` where the note will be created. The path may
                     contain spaces, and the last segment of the path will
                     be the note title.
        :param markdown_content: The content of the note in Markdown format
        :return: The ID of newly created note
        :raises SiYuanApiError: If the request fails
        """
        response = await self._raw_post(
            url='/api/filetree/createDoc',
            payload={
                'notebook': notebook_id,
                'path': path,
                'markdown': markdown_content,
            },
        )

        LOGGER.info('Note created in path %s', path)

        return response

    async def insert_block(self,
                           markdown_content: str,
                           next_id: str = None,
                           previous_id: str = None,
                           parent_id: str = None,
                           ) -> str:
        """
        Insert a block into a note by locators

        At least one of the locators need to be present. If multiple are present,
        the priority is: next_id > previous_id > parent_id.
        :param markdown_content: The markdown content of the block. If the
            content is complex, multiple blocks may be created
        :param next_id: The block ID AFTER the intended insertion location
        :param previous_id: The block ID BEFORE the intended insertion location
        :param parent_id: The ID of the parent block. The new block will be placed
            at the last of its children
        :return: The ID of the block created
        :raises SiYuanApiError: If the request fails
        """
        if not next_id and not previous_id and not parent_id:
            raise SiYuanApiError(
                message='At least one of next_id, previous_id or parent_id must be provided',
            )

        response = await self._raw_post(
            url='/api/block/insertBlock',
            payload={
                'dataType': 'markdown',
                'data': markdown_content,
                'nextID': next_id or '',
                'previousID': previous_id or '',
                'parentID': parent_id or '',
            },
        )

        if not isinstance(response, list):
            LOGGER.error(
                'Unexpected response type from API: %s',
                type(response),
            )

            raise SiYuanApiError(
                message='Unexpected response type from API',
            )

        if response[0]['undoOperations']:
            LOGGER.error(
                'Failed to insert block: %s',
                response[0]['undoOperations'],
            )

            raise SiYuanApiError(
                message='Operation failed'
            )

        LOGGER.info('Block inserted')
        LOGGER.debug('Block inserted: %s', response[0])

        return response[0]['doOperations']['id']

    async def set_block_attribute(self,
                                  block_id: str,
                                  attributes: dict[str, str],
                                  ):
        """
        Set attributes of an existing block

        :param block_id: The ID of the block to set attributes for
        :param attributes: A dictionary of attributes to set for the block
        :raises SiYuanApiError: If the request fails
        """
        await self._raw_post(
            url='/api/attr/setBlockAttrs',
            payload={
                'id': block_id,
                'attrs': attributes,
            }
        )

        LOGGER.info('Block attributes set for block %s', block_id)
