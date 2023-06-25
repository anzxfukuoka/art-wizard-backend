import cv2
import math
import numpy as np


# class Point:
#     """
#     2D point
#     """
#
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def __repr__(self):
#         return "Point({}, {})".format(self.x, self.y)


class Rect:
    """
    2D Rect
    x, y - top left point
    w, h - rect sizes
    """

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return "Rect({}, {}, {}, {})".format(self.x, self.y, self.w, self.h)


def crop(image: np.ndarray, x: int, y: int, w: int, h: int):
    """
    :param image: cv2 image
    :param x: top left x
    :param y: top left y
    :param w: crop width
    :param h: crop height
    :return: cropped image
    """
    cropped = image.copy()
    cropped = cropped[y: y + h, x: x + w]
    return cropped


def distance(pt1, pt2):
    """
    :param pt1: first point: tuple(x, y)
    :param pt2: second point: tuple(x, y)
    :return: distance between pt1 and pt2
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


#
# def get_edges(image: np.array):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image_grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
#     image_grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
#     image_edges = cv2.addWeighted(image_grad_x, 1, image_grad_y, 1, 0)
#     image_edges = image_edges.astype(np.float32)
#
#     return image_edges


def resize_fill(image: np.ndarray, size: tuple):
    """
    :param image: cv2 image
    :param size: tuple(image width, image height)
    :return:
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
