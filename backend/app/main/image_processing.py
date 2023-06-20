# from imutils.object_detection import non_max_suppression
# from google.colab.patches import cv2_imshow
import numpy as np
import argparse
import time
import cv2

import pytesseract

import re
import math


def extract_art_data(image: np.ndarray):
    # east: image height and width should be multiple of 32
    pp_image = preprocess_image(image)

    return pp_image.shape

def preprocess_image(image: np.array):
    result = resize(image, (640, 640))
    return result


def resize(image: np.ndarray, size: tuple):
    """
    img -> cv2 image
    size -> tuple(x, y)
    """
    h, w = image.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:
        interp = cv2.INTER_AREA  # shrinking image
    else:
        interp = cv2.INTER_CUBIC  # stretching image

    # aspect ratio of image
    aspect = w / h

    # compute scaling and pad sizing
    if aspect > 1:
        # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # scale and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT)

    return scaled_img
