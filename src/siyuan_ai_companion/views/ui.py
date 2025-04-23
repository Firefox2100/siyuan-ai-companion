import importlib.resources as pkg_resources
from pathlib import Path
from quart import Blueprint, send_from_directory, redirect, url_for


ui_blueprint = Blueprint('ui', __name__)


@ui_blueprint.route('/<path:filename>', methods=['GET'])
async def serve_ui(filename: str):
    try:
        filename = filename.lstrip('/')

        # Check if the file exists in the package resources
        with pkg_resources.path('siyuan_ai_companion.data', 'ui') as base_path:
            full_path = Path(base_path) / filename
            if not full_path.exists() or not full_path.is_file():
                raise FileNotFoundError

            return await send_from_directory(base_path, filename)
    except FileNotFoundError:
        # If the file is not found, return a 404 error
        return {'error': 'File not found'}, 404


@ui_blueprint.route('/', methods=['GET'])
async def ui_redirect():
    """
    Redirect to the main UI page.
    """
    return redirect(url_for('.serve_ui', filename='index.html'))
