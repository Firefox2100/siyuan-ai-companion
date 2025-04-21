from unittest.mock import AsyncMock
from qdrant_client.http.models import ScoredPoint
from siyuan_ai_companion.model.rag_driver import RagDriver


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
        driver.add_block('block1', 'doc1', 'test content')

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
        driver.add_blocks([
            ('block1', 'test content 1', 'doc1'),
            ('block2', 'test content 2', 'doc2')
        ])

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
        driver.update_block('block1', 'doc1', 'updated content')

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
        driver.update_blocks([
            ('block1', 'updated content 1', 'doc1'),
            ('block2', 'updated content 2', 'doc2')
        ])

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
            ScoredPoint(
                id=1,
                score=0.9,
                payload={'blockId': 'block1', 'documentId': 'doc1', 'content': 'test content'},
                version=0
            )
        ]
        mock_client.query_points.return_value = mock_hits
        mocker.patch.object(RagDriver, 'client', mock_client)
        mocker.patch.object(RagDriver, 'transformer', mock_transformer)

        driver = RagDriver()
        results = driver.search('query')

        assert results == [{
            'blockId': 'block1',
            'documentId': 'doc1',
            'content': 'test content',
            'score': 0.9
        }]

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
            {'blockId': 'block1', 'documentId': 'doc1', 'content': 'Block 1 content'},
            {'blockId': 'block2', 'documentId': 'doc2', 'content': 'Block 2 content'},
        ])

        # Mock SiyuanApi async context manager
        mock_siyuan = mocker.AsyncMock()
        mock_siyuan.get_note_markdown.side_effect = [
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

    def test_segment_document_with_headers(self, mocker):
        driver = RagDriver()

        # Provide a large enough token size to force segmentation
        mocker.patch.object(driver, '_estimate_tokens', return_value=1000)

        document = "# Header 1\nContent 1\n\n## Header 2\nContent 2"
        matching_blocks = ["Content 1", "Content 2"]

        segments = driver._segment_document(document, matching_blocks)
        assert len(segments) == 2, segments
        assert "Content 1" in segments[0]
        assert "Content 2" in segments[1]

    def test_segment_document_without_headers(self, mocker):
        driver = RagDriver()

        # Provide a large enough token size to force segmentation
        mocker.patch.object(driver, '_estimate_tokens', return_value=1000)

        document = "Paragraph 1\n\nParagraph 2"
        matching_blocks = ["Paragraph 1"]

        segments = driver._segment_document(document, matching_blocks)
        assert len(segments) == 1, segments
        assert "Paragraph 1" in segments[0]

    def test_fallback_split(self, mocker):
        driver = RagDriver()
        mocker.patch.object(driver, '_estimate_tokens', side_effect=lambda x: len(x.split()))

        document = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        segments = driver._fallback_split(document, max_tokens=5)
        assert len(segments) == 2, segments
        assert "Paragraph 1" in segments[0]
        assert "Paragraph 2" in segments[0]
        assert "Paragraph 3" in segments[1]

    async def test_get_context(self, mocker):
        driver = RagDriver()

        # Patch the search method to return mock search results
        mocker.patch.object(driver, 'search', return_value=[
            {'blockId': 'block1', 'documentId': 'doc1', 'content': 'Content 1'},
            {'blockId': 'block2', 'documentId': 'doc2', 'content': 'Content 2'}
        ])

        # Create a mock SiyuanApi context manager
        mock_siyuan_instance = AsyncMock()
        mock_siyuan_instance.get_note_markdown.side_effect = [
            "# Doc 1\nContent 1", "# Doc 2\nContent 2"
        ]

        mock_siyuan_cm = AsyncMock()
        mock_siyuan_cm.__aenter__.return_value = mock_siyuan_instance
        mock_siyuan_cm.__aexit__.return_value = None

        # Patch the SiyuanApi class to return the mock context manager
        mocker.patch(
            'siyuan_ai_companion.model.rag_driver.SiyuanApi',
            return_value=mock_siyuan_cm
        )

        context = await driver.get_context("query", limit=2)
        assert len(context) == 2

    async def test_build_prompt(self, mocker):
        driver = RagDriver()
        mocker.patch.object(driver, 'get_context', return_value=[
            "Context 1",
            "Context 2"
        ])

        prompt = await driver.build_prompt("What is AI?", limit=2)
        assert "Context 1" in prompt
        assert "Context 2" in prompt
        assert "What is AI?" in prompt
