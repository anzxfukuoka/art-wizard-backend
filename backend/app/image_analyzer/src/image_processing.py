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

        data_folder = os.path.join(os.path.dirname(image_analyzer.__file__), "./data/")

        # load EAST in Network
        self.EASTnet = cv2.dnn.readNet(os.path.join(data_folder, 'frozen_east_text_detection.pb'))

        # load templates
        self.star_template = cv2.imread(os.path.join(data_folder, 'star_template.png'), cv2.IMREAD_COLOR)

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

        image_x, image_y = (0, 0)
        image_h, image_w = preprocessed_image.shape[:2]

        image_crop_upper = crop(preprocessed_image, image_x, image_y, image_w, star_y)
        image_crop_middle = crop(preprocessed_image, image_x, star_y, image_w, star_yh - star_y)
        image_crop_lower = crop(preprocessed_image, image_x, star_yh, image_w, image_h - star_yh)

        # Text recognition

        image_crop_upper = resize_fill(image_crop_upper, (self.image_size, self.image_size))
        image_crop_lower = resize_fill(image_crop_lower, (self.image_size, self.image_size))

        # show the output image
        cv2.imshow("", image_crop_lower)
        cv2.waitKey(0)

        bounds = self._find_text_bounds(image_crop_lower)

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in bounds:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX)
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)

            # draw the bounding box on the image
            cv2.rectangle(image_crop_lower, (startX, startY), (endX, endY), (255, 255, 255), 2)

        # show the output image
        cv2.imshow("", image_crop_lower)
        cv2.waitKey(0)

        return None

    def _preprocess_image(self, image: np.ndarray, image_size, brightness=-60, contrast=1.5):

        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        resized = resize_fill(adjusted, (image_size, image_size))
        #grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(resized, (3, 3), 0)

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
        W, H = template.shape[:2]

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

    def _find_text_bounds(self, image: np.ndarray):
        # Create Blob from Image
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Add layers for network
        outputLayers = []
        outputLayers.append("feature_fusion/Conv_7/Sigmoid")
        outputLayers.append("feature_fusion/concat_3")

        # Pass Input to Network and get the Ouput based on layers
        self.EASTnet.setInput(blob)
        scores, geometry = self.EASTnet.forward(outputLayers)

        # Get rects and confidence score for bounding boxes
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        return boxes


if __name__ == '__main__':
    path = os.path.join(r"C:\MISC\Dev\py\genshin_tool_3.0\backend\app\image_analyzer\tests\data\img.png")
    timg = cv2.imread(path, cv2.IMREAD_COLOR)

    extractor = ArtExtractor()
    result = extractor.art_from_image(timg)
