"""
Blueprint definition for OpenAI compatible API endpoints.
"""

from urllib.parse import urljoin
from quart import Blueprint, request, jsonify

from siyuan_ai_companion.consts import APP_CONFIG
from siyuan_ai_companion.model import RagDriver
from .utils import token_required, forward_request


openai_blueprint = Blueprint('openai', __name__)


@openai_blueprint.route('/rag/v1/chat/completions', methods=['POST'])
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
        chat_model = request_payload.pop('tokenizerModel')
    else:
        chat_model = request_payload.get('model')

    for message in request_payload.get('messages', []):
        if message.get('role') == 'user':
            user_message = message.get('content', '')
            break

    if not user_message:
        return jsonify({'error': 'No user message provided'}), 400

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

    target_url = urljoin(APP_CONFIG.openai_url, '/chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/direct/v1/chat/completions', methods=['POST'])
@token_required
async def v1_chat_completion_direct():
    """
    Direct proxy to OpenAI chat completion endpoint.

    The prompt in this endpoint is expected to be a list of messages.
    """
    request_payload = await request.get_json()
    target_url = urljoin(APP_CONFIG.openai_url, '/chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/rag/v1/completions', methods=['POST'])
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
        return jsonify({'error': 'No prompt provided'}), 400

    if 'tokenizerModel' in request_payload:
        # Using a third-party model runner like ollama, which does not
        # correspond to huggingface models
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
    target_url = urljoin(APP_CONFIG.openai_url, '/embeddings')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/rag/v1//models', methods=['GET'])
@openai_blueprint.route('/direct/v1//models', methods=['GET'])
@token_required
async def v1_models():
    """
    Handles the /models endpoint.

    This endpoint returns a model list from the OpenAI API.
    """
    target_url = urljoin(APP_CONFIG.openai_url, '/models')
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
        return jsonify({'error': 'No user message provided'}), 400

    rag_driver = RagDriver()
    rag_driver.selected_model = chat_model
    context = await rag_driver.get_context(
        query=user_message,
    )

    return jsonify({'context': context})
