import cv2
import os
from app.image_analyzer import tests
import jsonpickle

def test_status_code(client):
    response = client.get("/image_analyzer/")
    assert response.status_code == 200


def test_get_art_data(client):
    # prepare headers for http request
    content_type = 'image/png'
    headers = {'content-type': content_type}

    path = os.path.join(os.path.dirname(tests.__file__), './data/img.png')
    img = cv2.imread(path)

    # encode image
    _, img_encoded = cv2.imencode('.png', img)

    response = client.post("/image_analyzer/get_art_data", data=img_encoded.tobytes())
    # assert b"<h2>Hello, World!</h2>" in response.data
    assert response.status_code == 200

    data = jsonpickle.decode(response.data)

    assert data["result"] == 5
