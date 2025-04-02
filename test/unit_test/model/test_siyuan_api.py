"""
Tests related to SiYuan API handling
"""

from datetime import datetime
import pytest
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

    def test_sorts_nodes_with_no_children(self):
        """
        Sorting root nodes without children
        """
        nodes = [
            {'id': '1', 'parent_id': None, 'sort': 2},
            {'id': '2', 'parent_id': None, 'sort': 1},
        ]
        sorted_nodes = SiyuanApi._sort_nodes(nodes)
        assert sorted_nodes == [
            {'id': '2', 'parent_id': None, 'sort': 1},
            {'id': '1', 'parent_id': None, 'sort': 2},
        ]

    def test_sorts_nodes_with_children(self):
        """
        Note 2 has 2 children
        """
        nodes = [
            {'id': '1', 'parent_id': None, 'sort': 2},
            {'id': '2', 'parent_id': None, 'sort': 1},
            {'id': '3', 'parent_id': '2', 'sort': 1},
            {'id': '4', 'parent_id': '2', 'sort': 2},
        ]
        sorted_nodes = SiyuanApi._sort_nodes(nodes)
        assert sorted_nodes == [
            {'id': '2', 'parent_id': None, 'sort': 1},
            {'id': '3', 'parent_id': '2', 'sort': 1},
            {'id': '4', 'parent_id': '2', 'sort': 2},
            {'id': '1', 'parent_id': None, 'sort': 2},
        ]

    def test_sorts_nodes_with_multiple_levels(self):
        """
        Note 2 has 1 child, and note 3 has 1 child
        """
        nodes = [
            {'id': '1', 'parent_id': None, 'sort': 2},
            {'id': '2', 'parent_id': None, 'sort': 1},
            {'id': '3', 'parent_id': '2', 'sort': 1},
            {'id': '4', 'parent_id': '3', 'sort': 1},
        ]
        sorted_nodes = SiyuanApi._sort_nodes(nodes)
        assert sorted_nodes == [
            {'id': '2', 'parent_id': None, 'sort': 1},
            {'id': '3', 'parent_id': '2', 'sort': 1},
            {'id': '4', 'parent_id': '3', 'sort': 1},
            {'id': '1', 'parent_id': None, 'sort': 2},
        ]

    def test_sorts_nodes_with_same_sort_value(self):
        """
        Sorting nodes with the same sort value

        This should not happen in practice, but just in case
        it should not raise an error either
        """
        nodes = [
            {'id': '1', 'parent_id': None, 'sort': 1},
            {'id': '2', 'parent_id': None, 'sort': 1},
        ]
        sorted_nodes = SiyuanApi._sort_nodes(nodes)
        assert sorted_nodes == [
            {'id': '1', 'parent_id': None, 'sort': 1},
            {'id': '2', 'parent_id': None, 'sort': 1},
        ]

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

    async def test_retrieves_blocks_by_note_id(self, mocker):
        """
        Retrieve multiple blocks by note ID
        """
        mocker.patch.object(
            SiyuanApi,
            '_raw_query',
            return_value=[{'id': 'block1', 'parent_id': '', 'root_id': 'note1', 'sort': 1}]
        )
        mocker.patch.object(
            SiyuanApi,
            'get_count',
            return_value=1
        )
        api = SiyuanApi()
        api._block_count = 1
        result = await api.get_blocks_by_note('note1')
        assert result == [{'id': 'block1', 'parent_id': '', 'root_id': 'note1', 'sort': 1}]

    async def test_retrieves_note_plaintext(self, mocker):
        """
        Retrieve the plaintext content of a note
        """
        mocker.patch.object(
            SiyuanApi,
            'get_blocks_by_note',
            return_value=[{'content': 'line1'}, {'content': 'line2'}]
        )
        api = SiyuanApi()
        result = await api.get_note_plaintext('note1')
        assert result == 'line1\nline2'
