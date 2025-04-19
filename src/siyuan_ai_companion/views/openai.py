"""
Blueprint definition for OpenAI compatible API endpoints.
"""

from urllib.parse import urljoin
from quart import Blueprint, request, jsonify

from siyuan_ai_companion.consts import APP_CONFIG
from siyuan_ai_companion.model import RagDriver
from .utils import token_required, forward_request


openai_blueprint = Blueprint('openai', __name__)


@openai_blueprint.route('/v1/chat/completions', methods=['POST'])
@token_required
async def v1_chat_completion():
    """
    Handles the /v1/chat/completions endpoint.

    The prompt in this endpoint is expected to be a list of messages.
    """
    request_payload = await request.get_json()

    user_message = ''
    for message in request_payload.get('messages', []):
        if message.get('role') == 'user':
            user_message = message.get('content', '')
            break

    if not user_message:
        return jsonify({'error': 'No user message provided'}), 400

    rag_driver = RagDriver()

    new_prompt = await rag_driver.build_prompt(query=user_message)

    # Inject the RAG-generated prompt into the user message
    for message in request_payload['messages']:
        if message['role'] == 'user':
            message['content'] = new_prompt
            break

    target_url = urljoin(APP_CONFIG.openai_url, '/chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/v1/completions', methods=['POST'])
@token_required
async def v1_completions():
    """
    Handles the /v1/completions endpoint.

    This is raw completion endpoint which will use the prompt directly with the model,
    the prompt is expected in `prompt` field of the payload.
    """
    request_payload = await request.get_json()

    prompt = request_payload.get("prompt")
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    rag_driver = RagDriver()
    new_prompt = await rag_driver.build_prompt(query=prompt)
    request_payload["prompt"] = new_prompt

    target_url = urljoin(APP_CONFIG.openai_url, 'completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/v1/embeddings', methods=['POST'])
@token_required
async def v1_embeddings():
    """
    Handles the /v1/embeddings endpoint.

    This uses the model embedding directly. No prompts injected
    """
    request_payload = await request.get_json()
    target_url = urljoin(APP_CONFIG.openai_url, '/embeddings')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/models', methods=['GET'])
@token_required
async def v1_models():
    """
    Handles the /models endpoint.

    This endpoint returns a model list from the OpenAI API.
    """
    target_url = urljoin(APP_CONFIG.openai_url, '/models')
    return await forward_request(target_url, None, method='GET')
