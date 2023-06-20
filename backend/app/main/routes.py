from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

from app.main import bp
from app.main.image_processing import extract_art_data

@bp.route('/')
def index():
    return 'This is The Main Blueprint'


@bp.route('/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    result = extract_art_data(img)

    # build a response dict to send back to client
    response = {'message': 'image processed. size={}x{}'.format(result[1], result[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")
