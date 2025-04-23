"""
Main application entry point for the Siyuan AI Companion.

In debug mode, run a Hypercorn debug server with CORS disabled.
DO NOT USE THIS IN PRODUCTION.
"""

import os
import logging
from quart import Quart, redirect, url_for
from quart_cors import cors
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from siyuan_ai_companion.tasks import update_index
from siyuan_ai_companion.views import asset_blueprint, openai_blueprint, \
    ui_blueprint


def create_app(debug = False):
    """
    Create and configure the Quart application.
    """
    quart_app = Quart(__name__)

    if debug:
        cors(                           # Disable CORS for all routes
            quart_app,
            allow_origin="*",
            allow_headers=["*"],
            allow_methods=["*"],
        )
        quart_app.logger.setLevel(logging.DEBUG)    # Set logging level to DEBUG

    quart_app.register_blueprint(asset_blueprint, url_prefix='/assets')
    quart_app.register_blueprint(openai_blueprint, url_prefix='/openai')
    quart_app.register_blueprint(ui_blueprint, url_prefix='/ui')

    # Initialize the scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(update_index, 'interval', minutes=5)

    @quart_app.before_serving
    async def startup():
        LOGGER.info('Beginning application startup')
        # Start the scheduler when the app starts
        scheduler.start()

        # Run the task immediately
        if APP_CONFIG.force_update_index:
            # Remove the timestamp file
            try:
                os.remove('last_update')
                LOGGER.info('Last update timestamp cleared, forcing index update')
            except FileNotFoundError:
                pass

        await update_index()

        LOGGER.info('Application startup complete. Ready to serve requests.')

    @quart_app.route('/health')
    async def health_check():
        return {'status': 'healthy'}

    @quart_app.route('/')
    async def redirect_to_ui():
        """
        Redirect to the main UI page.
        """
        return redirect(url_for('ui.serve_ui', filename='index.html'))

    return quart_app


if __name__ == '__main__':
    app = create_app(
        debug=True
    )
    app.run()
