"""
Blueprint definition for OpenAI compatible API endpoints.
"""

from urllib.parse import urljoin
from quart import Blueprint, Response, request, jsonify
import httpx

from siyuan_ai_companion.consts import OPENAI_URL
from siyuan_ai_companion.model import RagDriver


openai_blueprint = Blueprint('openai', __name__)


async def forward_request(url: str,
                          payload: dict | None,
                          method='POST',
                          ) -> tuple[str, int, list[tuple[str, str]]] | Response:
    """
    Forwards the request to the OpenAI API and returns the response.

    Stream responses are handled differently based on the method and payload.
    :param url: The URL to forward the request to.
    :param payload: The payload to send in the request.
    :param method: The HTTP method to use (default is POST).
    :return: The response from the OpenAI API. If streaming response is enabled,
             it returns a Response object for streaming. Otherwise it unpacks the
             response into Quart handler response format.
    """
    headers = {
        'Authorization': request.headers.get('Authorization', ''),
        'Content-Type': 'application/json',
        'User-Agent': request.headers.get('User-Agent', 'SiYuanApiCompanion/1.0'),
        'Accept': request.headers.get('Accept', 'application/json'),
        'Host': request.headers.get('Host', ''),
    }

    # Detect if client wants a streamed response
    stream = False
    if method == 'POST' and isinstance(payload, dict):
        stream = payload.get("stream", False)

    if stream:
        async def async_stream():
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(method, url, json=payload, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return Response(async_stream(), content_type='text/event-stream')

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=url,
            json=payload if method == 'POST' else None,
            headers=headers,
            timeout=30.0,
        )

    return response.text, response.status_code, response.headers.items()


@openai_blueprint.route('/v1/chat/completions', methods=['POST'])
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

    target_url = urljoin(OPENAI_URL, '/chat/completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/v1/completions', methods=['POST'])
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

    target_url = urljoin(OPENAI_URL, 'completions')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/v1/embeddings', methods=['POST'])
async def v1_embeddings():
    """
    Handles the /v1/embeddings endpoint.

    This uses the model embedding directly. No prompt injected
    """
    request_payload = await request.get_json()
    target_url = urljoin(OPENAI_URL, '/embeddings')
    return await forward_request(target_url, request_payload)


@openai_blueprint.route('/models', methods=['GET'])
async def v1_models():
    """
    Handles the /models endpoint.

    This endpoint returns a model list from the OpenAI API.
    """
    target_url = urljoin(OPENAI_URL, '/models')
    return await forward_request(target_url, None, method='GET')
