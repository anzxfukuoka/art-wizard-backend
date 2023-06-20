# from imutils.object_detection import non_max_suppression
# from google.colab.patches import cv2_imshow
import numpy as np
import argparse
import time
import cv2
import os

from app import image_analyzer
import re
import math

dir = os.path.dirname(image_analyzer.__file__)
# load templates
star_template = cv2.imread(os.path.join(dir, './data/star_template.png'), cv2.IMREAD_COLOR)


def extract_art_data(image: np.ndarray):
    # east: image height and width should be multiple of 32
    pp_image = _preprocess_image(image)

    return pp_image


def _preprocess_image(image: np.array):
    resized = _resize(image, (640, 640))
    star_template_gray = cv2.cvtColor(star_template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = _find_template(image_gray, star_template_gray)
    return result


def _find_template(image: np.array, star_template: np.array):
    search_result = cv2.matchTemplate(image, star_template, cv2.TM_CCOEFF_NORMED)

    # h, w, channels = star_template.shape
    h, w = star_template.shape

    threshold = 0.4
    doublicates_radius_threshold = w * 0.6

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_result)

    loc = np.where(search_result >= threshold)

    tst_image_copy = image.copy()

    # found template
    # doubles detection
    templates_points = []
    found_templates = []

    def distance(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    for pt in zip(*loc[::-1]):
        found_pt = True

        # removing doubles
        for template_pt in templates_points:
            if distance(pt, template_pt) < doublicates_radius_threshold:
                found_pt = False
                break

        if found_pt:
            templates_points.append(pt)
            top_left = pt
            bottom_right = (pt[0] + w, pt[1] + h)
            cv2.rectangle(tst_image_copy, top_left, bottom_right, (255, 255, 255), 2)
            found_templates.append((top_left, bottom_right, w, h))

    print("count", len(found_templates), len(loc[0]))
    return len(found_templates)


def _get_edges(image: np.array):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    image_grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    image_edges = cv2.addWeighted(image_grad_x, 1, image_grad_y, 1, 0)
    image_edges = image_edges.astype(np.float32)

    return image_edges


def _resize(image: np.ndarray, size: tuple):
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
