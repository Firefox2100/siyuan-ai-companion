import os
from quart import Quart
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from siyuan_ai_companion.consts import FORCE_UPDATE_INDEX
from siyuan_ai_companion.tasks import update_index
from siyuan_ai_companion.views import v1_blueprint


def create_app():
    quart_app = Quart(__name__)

    quart_app.register_blueprint(v1_blueprint, url_prefix='/v1')

    # Initialize the scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(update_index, 'interval', minutes=5)

    @quart_app.before_serving
    async def startup():
        # Start the scheduler when the app starts
        scheduler.start()

        # Run the task immediately
        if FORCE_UPDATE_INDEX:
            # Remove the timestamp file
            try:
                os.remove('last_update')
            except FileNotFoundError:
                pass

        await update_index()

    @quart_app.route('/health')
    async def health_check():
        return {'status': 'healthy'}

    return quart_app


if __name__ == '__main__':
    app = create_app()
    app.run()
