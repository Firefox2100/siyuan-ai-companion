from urllib.parse import urljoin
from quart import Blueprint, request, jsonify
import httpx

from siyuan_ai_companion.consts import OPENAI_URL
from siyuan_ai_companion.model import RagDriver


v1_blueprint = Blueprint('v1', __name__)


async def forward_request(url, payload, method='POST'):
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=url,
            json=payload if method == 'POST' else None,
            headers={
                "Authorization": request.headers.get("Authorization", ""),
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    return response.text, response.status_code, response.headers.items()


@v1_blueprint.route('/chat/completions', methods=['POST'])
async def chat_completion():
    request_payload = await request.get_json()

    user_message = ''
    for message in request_payload.get('messages', []):
        if message.get('role') == 'user':
            user_message = message.get('content', '')
            break

    if not user_message:
        return jsonify({'error': 'No user message provided'}), 400

    rag_driver = RagDriver()

    new_prompt = rag_driver.build_prompt(query=user_message)

    # Inject the RAG-generated prompt into the user message
    for message in request_payload['messages']:
        if message['role'] == 'user':
            message['content'] = new_prompt
            break

    target_url = urljoin(OPENAI_URL, '/chat/completions')
    return await forward_request(target_url, request_payload)


@v1_blueprint.route('/completions', methods=['POST'])
async def completions():
    request_payload = await request.get_json()

    prompt = request_payload.get("prompt")
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    rag_driver = RagDriver()
    new_prompt = rag_driver.build_prompt(query=prompt)
    request_payload["prompt"] = new_prompt

    target_url = urljoin(OPENAI_URL, '/completions')
    return await forward_request(target_url, request_payload)


@v1_blueprint.route('/embeddings', methods=['POST'])
async def embeddings():
    request_payload = await request.get_json()
    target_url = urljoin(OPENAI_URL, '/embeddings')
    return await forward_request(target_url, request_payload)


@v1_blueprint.route('/models', methods=['GET'])
async def models():
    target_url = urljoin(OPENAI_URL, '/models')
    return await forward_request(target_url, None, method='GET')
