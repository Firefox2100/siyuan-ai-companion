import pytest
from datetime import datetime
from httpx import AsyncClient
from siyuan_ai_companion.model.siyuan_api import SiyuanApi
from siyuan_ai_companion.errors import SiYuanApiError


class TestSiyuanApi:
    """
    Test cases for SiyuanApi class
    """

    async def test_successful_sql_query(self, mocker):
        """
        Successful SQL query execution
        """
        mock_client = mocker.AsyncMock(spec=AsyncClient)
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 0, 'data': [{'COUNT(*)': 5}]}
        mock_client.post.return_value = mock_response

        api = SiyuanApi(client=mock_client)
        result = await api._raw_query("SELECT COUNT(*) FROM blocks")
        assert result == [{'COUNT(*)': 5}]

    async def test_failed_sql_query_with_status_code(self, mocker):
        """
        SQL query failed with a non-200 status code
        """
        mock_client = mocker.AsyncMock(spec=AsyncClient)
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'error': 'Internal Server Error'}
        mock_client.post.return_value = mock_response

        api = SiyuanApi(client=mock_client)
        with pytest.raises(SiYuanApiError):
            await api._raw_query("SELECT COUNT(*) FROM blocks")

    async def test_failed_sql_query_with_json_code(self, mocker):
        """
        SQL query failed with 200 but an error code in response JSON
        """
        mock_client = mocker.AsyncMock(spec=AsyncClient)
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'code': 1, 'msg': 'Error'}
        mock_client.post.return_value = mock_response

        api = SiyuanApi(client=mock_client)
        with pytest.raises(SiYuanApiError):
            await api._raw_query("SELECT COUNT(*) FROM blocks")

    async def test_retrieves_block_count(self, mocker):
        """
        Retrieves the block count from the database
        """
        mocker.patch.object(SiyuanApi, '_raw_query', return_value=[{'COUNT(*)': 5}])
        api = SiyuanApi()
        result = await api.get_count()
        assert result == 5

    async def test_retrieves_block_by_id(self, mocker):
        """
        Retrieves a single block by its ID
        """
        mocker.patch.object(
            SiyuanApi,
            '_raw_query',
            return_value=[{'id': 'block1', 'content': 'test content'}]
        )
        api = SiyuanApi()
        result = await api.get_block('block1')
        assert result == {'id': 'block1', 'content': 'test content'}

    async def test_retrieves_blocks_by_time(self, mocker):
        """
        Retrieve multiple blocks updated after a certain time
        """
        mocker.patch.object(
            SiyuanApi,
            '_raw_query',
            return_value=[{'id': 'block1', 'updated': '20230101000000'}]
        )
        mocker.patch.object(
            SiyuanApi,
            'get_count',
            return_value=1
        )
        api = SiyuanApi()
        api._block_count = 1
        result = await api.get_blocks_by_time(
            updated_after=datetime(2023, 1, 1)
        )
        assert result == [{'id': 'block1', 'updated': '20230101000000'}]

    async def test_list_assets(self, mocker):
        """
        List all assets with optional suffix filtering
        """
        mocker.patch.object(
            SiyuanApi,
            '_list_files_recursive',
            return_value=['/data/assets/file1.mp3', '/data/assets/file2.wav']
        )
        api = SiyuanApi()
        result = await api.list_assets(suffixes=['.mp3'])
        assert result == ['file1.mp3']

    async def test_create_note(self, mocker):
        """
        Create a note in a notebook
        """
        mocker.patch.object(
            SiyuanApi,
            '_raw_post',
            return_value='note_id_123'
        )
        api = SiyuanApi()
        result = await api.create_note(
            notebook_id='notebook1',
            path='/path/to/note',
            markdown_content='# Title\nContent'
        )
        assert result == 'note_id_123'

    async def test_insert_block(self, mocker):
        """
        Insert a block into a note
        """
        mocker.patch.object(
            SiyuanApi,
            '_raw_post',
            return_value=[{'doOperations': {'id': 'block_id_123'}, 'undoOperations': None}]
        )
        api = SiyuanApi()
        result = await api.insert_block(
            markdown_content='New block content',
            parent_id='parent_block_id'
        )
        assert result == 'block_id_123'

    async def test_set_block_attribute(self, mocker):
        """
        Set attributes for a block
        """
        mocker.patch.object(SiyuanApi, '_raw_post', return_value=None)
        api = SiyuanApi()
        await api.set_block_attribute(
            block_id='block_id_123',
            attributes={'key': 'value'}
        )

    async def test_download_asset(self, mocker):
        """
        Download an asset and store it as a temporary file
        """
        mock_response = mocker.Mock()
        mock_response.headers = {'Content-Type': 'audio/mpeg'}
        mock_response.content = b'fake audio content'
        mocker.patch.object(SiyuanApi, '_raw_post', return_value=mock_response)

        api = SiyuanApi()
        async with api.download_asset('path/to/asset.mp3') as temp_file:
            temp_file.seek(0)  # Ensure the file pointer is at the beginning
            assert temp_file.read() == b'fake audio content'
