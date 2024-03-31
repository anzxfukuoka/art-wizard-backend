from flask import Blueprint

bp = Blueprint('image_analyzer', __name__, url_prefix="/image_analyzer")

from app.image_analyzer.src import routes