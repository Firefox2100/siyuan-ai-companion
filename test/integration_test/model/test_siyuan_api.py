import os
import datetime
import pytest

from siyuan_ai_companion.model.siyuan_api import SiyuanApi, SiYuanApiError


class TestSiyuanApi:
    def test_init(self):
        siyuan = SiyuanApi()

        siyuan_url = os.getenv('SIYUAN_URL')
        siyuan_token = os.getenv('SIYUAN_TOKEN')

        assert siyuan.url == siyuan_url
        assert siyuan.token == siyuan_token
        assert siyuan._client.base_url == siyuan_url

        if siyuan_token is not None:
            assert siyuan._client.headers['Authorization'] == f'Token {siyuan_token}'

    def test_init_with_parameters(self):
        siyuan = SiyuanApi(
            url='https://example.com',
            token='test_token',
        )

        assert siyuan.url == 'https://example.com'
        assert siyuan.token == 'test_token'
        assert siyuan._client.base_url == 'https://example.com'
        assert siyuan._client.headers['Authorization'] == 'Token test_token'

    async def test_aenter(self):
        async with SiyuanApi() as siyuan:
            siyuan_url = os.getenv('SIYUAN_URL')
            siyuan_token = os.getenv('SIYUAN_TOKEN')

            assert siyuan.url == siyuan_url
            assert siyuan.token == siyuan_token
            assert siyuan._client.base_url == siyuan_url

            if siyuan_token is not None:
                assert siyuan._client.headers['Authorization'] == f'Token {siyuan_token}'

    async def test_close(self):
        siyuan = SiyuanApi()

        assert not siyuan._client.is_closed

        await siyuan.close()

        assert siyuan._client.is_closed

    async def test_raw_query(self):
        async with SiyuanApi() as siyuan:
            payload = await siyuan._raw_query(
                sql_query='SELECT COUNT(*) FROM blocks',
            )

            assert isinstance(payload[0]['COUNT(*)'], int)

            with pytest.raises(SiYuanApiError):
                await siyuan._raw_query(
                    sql_query='MALFORMED SQL QUERY',
                )

    async def test_get_count(self):
        async with SiyuanApi() as siyuan:
            count = await siyuan.get_count()

            assert isinstance(count, int)
            assert count == siyuan._block_count

    async def test_get_block(self):
        async with SiyuanApi() as siyuan:
            block_id = os.getenv('SIYUAN_TEST_BLOCK_ID')

            payload = await siyuan.get_block(
                block_id=block_id,
            )

            assert isinstance(payload, dict)
            assert payload['id'] == block_id

    async def test_get_blocks_by_time(self):
        async with SiyuanApi() as siyuan:
            all_blocks = await siyuan.get_blocks_by_time()

            assert isinstance(all_blocks, list)
            assert len(all_blocks) > 0

            partial_blocks = await siyuan.get_blocks_by_time(
                updated_after=datetime.datetime(2024, 1, 1)
            )

            assert isinstance(partial_blocks, list)
            assert len(partial_blocks) > 0

            assert len(all_blocks) > len(partial_blocks)

    async def test_get_blocks_by_note(self):
        async with SiyuanApi() as siyuan:
            note_id = os.getenv('SIYUAN_TEST_NOTE_ID')

            payload = await siyuan.get_blocks_by_note(
                note_id=note_id,
            )

            assert isinstance(payload, list)
            assert len(payload) > 0

            for block in payload:
                assert block['root_id'] == note_id

    async def test_get_note_plaintext(self):
        async with SiyuanApi() as siyuan:
            note_id = os.getenv('SIYUAN_TEST_NOTE_ID')

            plaintext = await siyuan.get_note_plaintext(
                note_id=note_id,
            )

            assert isinstance(plaintext, str)
            assert len(plaintext) > 0
