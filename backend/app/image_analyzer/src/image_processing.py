# from imutils.object_detection import non_max_suppression
# from google.colab.patches import cv2_imshow
from abc import ABC

import numpy as np
import argparse
import time
import cv2
import os
from imutils.object_detection import non_max_suppression

from app import image_analyzer
from app.image_analyzer.src.utils import *

import logging

logger = logging.getLogger(__name__)


class ArtExtractor:
    def __init__(self):
        # east model requires image height and width = multiple of 32
        self.image_size = 640
        self.star_template_size = 32

        # load templates
        data_folder = os.path.dirname(image_analyzer.__file__)
        self.star_template = cv2.imread(os.path.join(data_folder, './data/star_template.png'), cv2.IMREAD_COLOR)

        # templates preprocessing
        self.star_template = self._preprocess_image(self.star_template, self.star_template_size)

    def art_from_image(self, image: np.ndarray):

        # image preprocessing
        preprocessed_image = self._preprocess_image(image, self.image_size)

        # Stars detection

        found_stars = self._find_template(preprocessed_image, self.star_template)
        stars_count = len(found_stars)

        if stars_count < 3 or stars_count > 5:
            logger.error("unexpected stars count")

        if stars_count == 0:
            raise Exception("No stars detected")

        # Divide main and sub stats images

        star_x, star_y, star_xw, star_yh = found_stars[0]

        # orig_h, orig_w, orig_channels = tst_image.shape

        image_x, image_y = (0, 0)
        image_h, image_w = preprocessed_image.shape

        image_crop_upper = crop(preprocessed_image, image_x, image_y, image_w, star_y)
        image_crop_middle = crop(preprocessed_image, image_x, star_y, image_w, star_yh - star_y)
        image_crop_lower = crop(preprocessed_image, image_x, star_yh, image_w, image_h - star_yh)

        return None

    def _preprocess_image(self, image: np.ndarray, image_size):

        resized = resize_image(image, (image_size, image_size))
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayscale, (3, 3), 0)

        return blur

    def _find_template(self, image: np.array, template: np.array, detection_threshold=0.82, overlap_threshold=.4):
        """
        :param image: cv2 image
        :param template: cv2 image
        :param detection_threshold: template detection threshold
        :param overlap_threshold: multiple objects detection threshold
        :return: a list of sets, each of which represents the rect of the found template: (top_left, bottom_right, w, h)
        """
        match = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Select rectangles with
        # confidence greater than threshold
        (y_points, x_points) = np.where(match >= detection_threshold)
        W, H = template.shape

        # initialize our list of bounding boxes
        boxes = list()

        # store co-ordinates of each bounding box
        # we'll create a new list by looping
        # through each pair of points
        for (x, y) in zip(x_points, y_points):
            # update our list of boxes
            boxes.append((x, y, x + W, y + H))

        # apply non-maxima suppression to the rectangles
        # this will create a single bounding box
        # for each object
        boxes = non_max_suppression(np.array(boxes), overlapThresh=overlap_threshold)

        ''' draw boxes
        # loop over the final bounding boxes
        for (x1, y1, x2, y2) in boxes:
            # draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          (255, 0, 0), 3)
        
        '''

        return boxes


if __name__ == '__main__':
    path = os.path.join(r"C:\MISC\Dev\py\genshin_tool_3.0\backend\app\image_analyzer\tests\data\img.png")
    timg = cv2.imread(path, cv2.IMREAD_COLOR)

    extractor = ArtExtractor()
    result = extractor.art_from_image(timg)
