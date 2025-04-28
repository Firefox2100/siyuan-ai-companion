"""
Blueprint definition for assets processing endpoints.
"""

import asyncio
from quart import Blueprint, request, jsonify

from siyuan_ai_companion.model import Transcriber, SiyuanApi
from .utils import token_required, error_handler


asset_blueprint = Blueprint('transcribe', __name__)


@asset_blueprint.route('/', methods=['GET'])
@error_handler
@token_required
async def get_assets():
    """
    Get the list of all assets from SiYuan server.
    :return:
    """
    async with SiyuanApi() as siyuan:
        suffixes = request.args.getlist('suffix')

        assets = await siyuan.list_assets(suffixes=suffixes)

    return jsonify({
        'assets': assets,
    })


@asset_blueprint.route('/audio', methods=['GET'])
@error_handler
@token_required
async def get_audio_assets():
    """
    Get the list of all audio assets from SiYuan server.
    :return:
    """
    async with SiyuanApi() as siyuan:
        audio_assets = await siyuan.list_assets(suffixes=['wav'])
        audio_blocks = await siyuan.get_audio_blocks(
            audio_names=audio_assets,
        )
        transcription_ids = await siyuan.get_audio_transcription_ids(
            audio_ids=list(audio_blocks.values()),
        )

        result = {}

        for audio_path in audio_blocks.keys():
            result[audio_path] = transcription_ids.get(audio_blocks[audio_path])

        return jsonify(result)


@asset_blueprint.route('/audio/transcribe', methods=['POST'])
@error_handler
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
