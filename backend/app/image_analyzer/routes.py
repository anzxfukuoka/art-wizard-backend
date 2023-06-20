from flask import Flask, request, Response
from flask import current_app
import logging
import jsonpickle
import numpy as np
import cv2

from app.image_analyzer import bp
from app.image_analyzer.image_processing import extract_art_data

LOG = logging.getLogger(__name__)

@bp.route('/')
def index():
    return 'This is The {} Blueprint'.format(__name__)


@bp.route('/get_art_data', methods=['POST'])
def get_art_data():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    result = extract_art_data(img)

    # build a response dict to send back to client
    response = {'message': 'image processed. result={}'.format(result)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")
