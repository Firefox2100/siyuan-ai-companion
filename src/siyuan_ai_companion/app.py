from quart import Quart

from .views import v1_blueprint


def create_app():
    app = Quart(__name__)

    app.register_blueprint(v1_blueprint, url_prefix='/v1')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()
