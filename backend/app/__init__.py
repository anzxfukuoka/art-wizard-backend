from flask import Flask
from config import Config

from app.image_analyzer import bp as image_analyze_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask extensions here

    # Register blueprints here

    app.register_blueprint(image_analyze_bp) #'/image_analyzer'

    @app.route('/api/')
    def test_page():
        return '<h1>Welcome</h1> <p>i\'ts api endpoint</p>'

    return app
