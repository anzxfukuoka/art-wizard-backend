from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask extensions here

    # Register blueprints here
    from app.main import bp as main_bp
    app.register_blueprint(main_bp, url_prefix='/main')

    @app.route('/api/')
    def test_page():
        return '<h1>Welcome</h1> <p>i\'ts api endpoint</p>'

    return app
