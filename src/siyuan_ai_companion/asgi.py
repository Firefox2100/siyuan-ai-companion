"""
ASGI server entry for production use. This script exposes a variable
named `application` that can be used by ASGI servers to run the app.
"""

from siyuan_ai_companion.app import create_app


application = create_app()
