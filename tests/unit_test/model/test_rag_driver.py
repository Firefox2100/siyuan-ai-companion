import pytest
from qdrant_client.http.models import ScoredPoint
from siyuan_ai_companion.model.rag_driver import RagDriver
from siyuan_ai_companion.model.siyuan_api import SiyuanApi


class TestRagDriver:
    def test_add_single_block_to_index(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        driver.add_block('block1', 'test content')

        mock_client.upsert.assert_called_once()

    def test_add_multiple_blocks_to_index(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        driver.add_blocks([('block1', 'test content 1'), ('block2', 'test content 2')])

        mock_client.upsert.assert_called_once()

    def test_update_single_block_in_index(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        driver.update_block('block1', 'updated content')

        mock_client.upsert.assert_called_once()

    def test_update_multiple_blocks_in_index(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        driver.update_blocks([('block1', 'updated content 1'), ('block2', 'updated content 2')])

        mock_client.upsert.assert_called_once()

    def test_delete_single_block_from_index(self, mocker):
        mock_client = mocker.Mock()
        mocker.patch.object(RagDriver, 'client', mock_client)

        driver = RagDriver()
        driver.delete_block('block1')

        mock_client.delete.assert_called_once()

    def test_delete_all_blocks_from_index(self, mocker):
        mock_client = mocker.Mock()
        mocker.patch.object(RagDriver, 'client', mock_client)

        driver = RagDriver()
        driver.delete_all()

        mock_client.delete_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()

    def test_search_for_relevant_blocks(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor
        mock_hits = mocker.Mock()
        mock_hits.points = [
            ScoredPoint(id=1, score=0.9, payload={'blockId': 'block1'}, version=0)
        ]
        mock_client.query_points.return_value = mock_hits
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        results = driver.search('query')

        assert results == [{'blockId': 'block1', 'score': 0.9}]

    async def test_build_prompt(self, mocker):
        mock_client = mocker.Mock()
        mock_transformer = mocker.Mock()
        mock_tensor = mocker.Mock()
        mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.encode.return_value = mock_tensor

        # Patch RagDriver internals
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        # Mock search to return synthetic results
        mock_driver = RagDriver()
        mock_driver.search = mocker.Mock(return_value=[
            {'blockId': 'block1'},
            {'blockId': 'block2'},
        ])

        # Mock SiyuanApi async context manager
        mock_siyuan = mocker.AsyncMock()
        mock_siyuan.get_note_plaintext.side_effect = [
            'Document 1 content.',
            'Document 2 content.',
        ]

        mock_siyuan_cm = mocker.AsyncMock()
        mock_siyuan_cm.__aenter__.return_value = mock_siyuan
        mock_siyuan_cm.__aexit__.return_value = None

        mocker.patch('siyuan_ai_companion.model.rag_driver.SiyuanApi', return_value=mock_siyuan_cm)

        result = await mock_driver.build_prompt('What is AI?', limit=2)

        assert 'Document 1 content.' in result
        assert 'Document 2 content.' in result
        assert 'What is AI?' in result
