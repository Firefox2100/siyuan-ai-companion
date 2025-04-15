"""
Blueprint definition for assets processing endpoints.
"""

import asyncio
from quart import Blueprint, request, jsonify

from siyuan_ai_companion.model import Transcriber, SiyuanApi
from .utils import token_required


asset_blueprint = Blueprint('transcribe', __name__)


@asset_blueprint.route('/', methods=['GET'])
@token_required
async def get_assets():
    """
    Get the list of all assets from SiYuan server.
    :return:
    """
    async with SiyuanApi() as siyuan:
        suffixes = request.args.getlist('suffix')

        assets = await siyuan.list_assets(suffixes=suffixes)

    return jsonify(assets)


@asset_blueprint.route('/audio', methods=['GET'])
@token_required
async def get_audio_assets():
    """
    Get the list of all audio assets from SiYuan server.
    :return:
    """
    async with SiyuanApi() as siyuan:
        audio_assets = await siyuan.list_assets(suffixes=['wav'])
        audio_blocks = await siyuan.get_audio_blocks(
            audio_assets=audio_assets,
        )
        transcription_ids = await siyuan.get_audio_transcription_ids(
            audio_ids=audio_blocks,
        )

        for audio_block in audio_blocks:
            if not transcription_ids.get(audio_block):
                transcription_ids[audio_block] = None

        return jsonify(transcription_ids)


@asset_blueprint.route('/transcribe', methods=['POST'])
@token_required
async def transcribe_asset_file():
    """
    Transcribe an asset audio file in SiYuan server.
    :return:
    """
    payload = await request.get_json()

    asset_path = payload['assetPath']
    notebook_id = payload.get('notebookId')
    base_path = payload.get('basePath')
    title = payload.get('title')

    transcriber = Transcriber()

    asyncio.create_task(transcriber.process_asset(
        asset_path=asset_path,
        title=title,
        t_notebook=notebook_id,
        t_base_path=base_path,
    ))

    return jsonify({'status': 'processing'}), 202
