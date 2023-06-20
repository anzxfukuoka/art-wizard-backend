import cv2


def test_status_code(client):
    response = client.get("/image_analyzer/")
    assert response.status_code == 200


def test_get_art_data(client):
    # prepare headers for http request
    content_type = 'image/png'
    headers = {'content-type': content_type}

    img = cv2.imread('tests/data/img.png')

    # encode image
    _, img_encoded = cv2.imencode('.png', img)

    response = client.post("/image_analyzer/get_art_data", data=img_encoded.tostring())
    # assert b"<h2>Hello, World!</h2>" in response.data
    assert response.status_code == 200
