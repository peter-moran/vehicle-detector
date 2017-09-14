#!/usr/bin/env python
"""
Locates cars in video and places bounding boxes around them.

Author: Peter Moran
Created: 9/14/2017
"""

import sys
from typing import Tuple, List, Iterable

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from train import CarFeatureVectorBuilder

Rectangle = Tuple[Tuple[int, int], Tuple[int, int]]
'''A pair of (x, y) vertices defining a rectangle.'''


def draw_rectangles(img, rectangles: Iterable[Rectangle], color=(0, 0, 255), thick=6):
    """ Draws rectangles on a copy of the given image. """
    imcopy = np.copy(img)
    for bbox in rectangles:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def get_all_windows(img, window_shape=(64, 64), overlap=(0.5, 0.5), x_range=None, y_range=None) -> List[Rectangle]:
    """ Returns a list of Rectangles of the given shape spaced over an image. """
    # Fix range values
    img_h, img_w = img.shape[0:2]
    x_range = [None, None] if x_range is None else x_range
    y_range = [None, None] if y_range is None else y_range
    y_start = y_range[0] if y_range[0] is not None else 0
    y_stop = y_range[1] if y_range[1] is not None else img_h
    x_start = x_range[0] if x_range[0] is not None else 0
    x_stop = x_range[1] if x_range[1] is not None else img_w

    # Build list of Rectangles
    window_list = []
    for y in range(y_start, y_stop, int(window_shape[0] * overlap[0])):
        for x in range(x_start, x_stop, int(window_shape[1] * overlap[1])):
            p_upper_left = (x, y)
            p_lower_right = (x + window_shape[1], y + window_shape[0])
            if p_lower_right[0] > img_w or p_lower_right[1] > img_h:
                break
            window_list.append((p_upper_left, p_lower_right))
    return window_list


def search_windows(img, windows: Iterable[Rectangle], clf, feature_vector_builder, desired_label=1):
    """ Classifies the content within each window and returns windows classified to the given label. """
    desired_windows = []
    for window in windows:
        # Extract features from within the window
        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = feature_vector_builder.get_features([window_img])

        # Classify
        classification = clf.predict(features)
        if classification == desired_label:
            desired_windows.append(window)
    return desired_windows


if __name__ == '__main__':
    # Load parameters
    argc = len(sys.argv)
    clf_savefile = sys.argv[1] if argc > 1 else './data/trained_classifier.pkl'
    scaler_savefile = sys.argv[2] if argc > 2 else './data/Xy_scaler.pkl'

    # Set up classifier and feature builder
    print("Loading classifier from '{}'.".format(clf_savefile))
    svc = joblib.load(clf_savefile)
    print("Loading scaler from '{}'.".format(scaler_savefile))
    scaler = joblib.load(scaler_savefile)
    fvb = CarFeatureVectorBuilder(mode='image', feature_scaler=scaler)

    image = mpimg.imread('./data/test_images/test7.jpg')
    windows = get_all_windows(image, window_shape=(120, 120), overlap=(0.5, 0.5), y_range=(400, None))

    print('Searching for cars in image...')
    hot_windows = search_windows(image, windows, svc, fvb)

    window_img = draw_rectangles(image, hot_windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()
