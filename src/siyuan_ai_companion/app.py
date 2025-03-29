from quart import Quart
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .tasks import update_index
from .views import v1_blueprint


def create_app():
    quart_app = Quart(__name__)

    quart_app.register_blueprint(v1_blueprint, url_prefix='/v1')

    # Initialize the scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(update_index, 'interval', minutes=5)
    scheduler.start()

    return quart_app


if __name__ == '__main__':
    app = create_app()
    app.run()
