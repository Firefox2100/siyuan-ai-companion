"""
Blueprint definition for OpenAI compatible API endpoints.
"""

import io
import asyncio
from urllib.parse import urljoin
from quart import Blueprint, Response, request, jsonify, stream_with_context

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from siyuan_ai_companion.model import RagDriver, Transcriber
from .utils import CompanionEndpointHandlerError, token_required, forward_request, \
    error_handler


openai_blueprint = Blueprint('openai', __name__)


@openai_blueprint.route('/rag/v1/chat/completions', methods=['POST'])
@error_handler
@token_required
async def v1_chat_completion_rag():
    """
    Chat completion with automatic RAG prompt injection.

    The prompt in this endpoint is expected to be a list of messages.
    """
    request_payload: dict = await request.get_json()

    user_message = ''

    if 'tokenizerModel' in request_payload:
        # Using a third-party model runner like ollama, which does not
        # correspond to huggingface models
        LOGGER.debug('Tokenizer model override: %s', request_payload['tokenizerModel'])
        chat_model = request_payload.pop('tokenizerModel')
    else:
        chat_model = request_payload.get('model')

    for message in request_payload.get('messages', []):
        if message.get('role') == 'user':
            user_message = message.get('content', '')
            break

    if not user_message:
        raise CompanionEndpointHandlerError('No user message provided', 400)

    rag_driver = RagDriver()
    rag_driver.selected_model = chat_model
    new_prompt = await rag_driver.build_prompt(
        query=user_message,
    )

    # Inject the RAG-generated prompt into the user message
    for i in reversed(range(len(request_payload['messages']))):
        if request_payload['messages'][i]['role'] == 'user':
            request_payload['messages'][i]['content'] = new_prompt
            break

    target_url = urljoin(APP_CONFIG.openai_url, 'chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/direct/v1/chat/completions', methods=['POST'])
@error_handler
@token_required
async def v1_chat_completion_direct():
    """
    Direct proxy to OpenAI chat completion endpoint.

    The prompt in this endpoint is expected to be a list of messages.
    """
    request_payload = await request.get_json()
    target_url = urljoin(APP_CONFIG.openai_url, 'chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/rag/v1/completions', methods=['POST'])
@error_handler
@token_required
async def v1_completions_rag():
    """
    Completion with automatic RAG prompt injection.

    This is raw completion endpoint which will use the prompt directly with the model,
    the prompt is expected in `prompt` field of the payload.
    """
    request_payload = await request.get_json()

    prompt = request_payload.get('prompt')
    if not prompt:
        raise CompanionEndpointHandlerError('No prompt provided', 400)

    if 'tokenizerModel' in request_payload:
        # Using a third-party model runner like ollama, which does not
        # correspond to huggingface models
        LOGGER.debug('Tokenizer model override: %s', request_payload['tokenizerModel'])
        chat_model = request_payload.pop('tokenizerModel')
    else:
        chat_model = request_payload.get('model')

    rag_driver = RagDriver()
    rag_driver.selected_model = chat_model
    new_prompt = await rag_driver.build_prompt(
        query=prompt,
    )
    request_payload['prompt'] = new_prompt

    target_url = urljoin(APP_CONFIG.openai_url, 'completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/direct/v1/completions', methods=['POST'])
@token_required
async def v1_completions_direct():
    """
    Direct proxy to OpenAI completion endpoint.

    This is raw completion endpoint which will use the prompt directly with the model,
    the prompt is expected in `prompt` field of the payload.
    """
    request_payload = await request.get_json()
    target_url = urljoin(APP_CONFIG.openai_url, 'completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/rag/v1/embeddings', methods=['POST'])
@openai_blueprint.route('/direct/v1/embeddings', methods=['POST'])
@token_required
async def v1_embeddings():
    """
    Embedding generation endpoint.

    This uses the model embedding directly. No prompts injected
    """
    request_payload = await request.get_json()
    target_url = urljoin(APP_CONFIG.openai_url, 'embeddings')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/rag/v1/models', methods=['GET'])
@openai_blueprint.route('/direct/v1/models', methods=['GET'])
@token_required
async def v1_models():
    """
    Handles the /models endpoint.

    This endpoint returns a model list from the OpenAI API.
    """
    target_url = urljoin(APP_CONFIG.openai_url, 'models')
    return await forward_request(target_url, None, method='GET')


@openai_blueprint.route('/direct/v1/retrieve', methods=['POST'])
@token_required
async def v1_retrieve():
    """
    Retrieve the context for a user prompt, so the frontend may handle RAG locally
    """
    request_payload = await request.get_json()
    user_message = request_payload.get('prompt')
    chat_model = request_payload.get('model')

    if not user_message:
        raise CompanionEndpointHandlerError('No user message provided', 400)

    rag_driver = RagDriver()
    rag_driver.selected_model = chat_model
    context = await rag_driver.get_context(
        query=user_message,
    )

    return jsonify({'context': context})


@openai_blueprint.route('/direct/v1/transcribe', methods=['POST'])
@token_required
async def v1_transcribe():
    """
    Receive an audio file, transcribe it and stream the result back.
    """
    request_files = await request.files

    if 'file' not in request_files:
        raise CompanionEndpointHandlerError('No file part in the request', 400)

    file = request_files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_bytes = file.read()
    audio_buffer = io.BytesIO(file_bytes)

    transcriber = Transcriber()

    @stream_with_context
    async def streaming_response():
        async_generator = transcriber.process_buffer(audio_buffer)

        try:
            while True:
                try:
                    text = await asyncio.wait_for(async_generator.__anext__(), timeout=60)
                    yield text
                except asyncio.TimeoutError:
                    raise CompanionEndpointHandlerError(
                        'Timeout while waiting for transcription chunk',
                        408,
                    )
                except StopAsyncIteration:
                    break
        finally:
            if hasattr(async_generator, 'aclose'):
                await async_generator.aclose()

    response = Response(
        streaming_response(),
        content_type='text/plain',
    )
    response.timeout = None     # Disable total timeout for the stream

    return response
