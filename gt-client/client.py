from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:1707'
test_url = addr + '/image_analyzer/get_art_data'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('data/images/img.png')

# encode image
_, img_encoded = cv2.imencode('.png', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
