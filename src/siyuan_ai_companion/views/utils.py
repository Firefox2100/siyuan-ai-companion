"""
Utility functions for Quart request handlers.
"""

from functools import wraps
from copy import deepcopy
from quart import Response, request, jsonify
import httpx

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from siyuan_ai_companion.errors import SiYuanAiCompanionError


class CompanionEndpointHandlerError(SiYuanAiCompanionError):
    """
    Custom error class for handling errors in the request handler.
    """
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


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
    headers = deepcopy(request.headers)

    if APP_CONFIG.openai_token:
        # Add the OpenAI token to the headers if it is set
        headers['Authorization'] = f'Bearer {APP_CONFIG.openai_token}'
    else:
        headers.pop('Authorization')

    # Detect if the client wants a streamed response
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


def token_required(f):
    """
    A wrapper function to validate the token sent with the request
    """
    @wraps(f)
    async def decorated(*args, **kwargs):
        if APP_CONFIG.companion_token is None:
            # No token set, authentication disabled
            return await f(*args, **kwargs)

        token_header = request.headers.get('Authorization', None)
        if not token_header:
            return jsonify({'error': 'Authorization header is missing'}), 401

        if token_header.split(' ')[1] != APP_CONFIG.companion_token:
            return jsonify({'error': 'Invalid companion token'}), 401

        return await f(*args, **kwargs)

    return decorated


def error_handler(f):
    """
    A wrapper function to handle errors in the request handler
    """
    @wraps(f)
    async def decorated(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except SiYuanAiCompanionError as e:
            LOGGER.error('Error: %s', e.message)
            return jsonify({'error': e.message}), e.status_code
        except Exception as e:
            LOGGER.error('Unexpected error: %s', str(e))
            return jsonify({'error': str(e)}), 500

    return decorated
