#!/usr/bin/env python
"""
Locates cars in video and places bounding boxes around them.

Author: Peter Moran
Created: 9/14/2017
"""

import sys
from glob import glob
from typing import Tuple, Iterable

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from train import CarFeatureVectorBuilder, get_hog_features, CARS

Rectangle = Tuple[Tuple[int, int], Tuple[int, int]]
'''A pair of (x, y) vertices defining a rectangle.'''


def draw_rectangles(img, rectangles: Iterable[Rectangle], color=(0, 0, 255), thick=6):
    """ Draws rectangles on a copy of the given image. """
    imcopy = np.copy(img)
    for bbox in rectangles:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def find_cars(img, clf, fvb: CarFeatureVectorBuilder, y_range, window_scale, window_overlap):
    img = np.copy(img)[y_range[0]:y_range[1], :, :]
    img_h, img_w = img.shape[:2]

    # Scale image to search for different size objects
    if window_scale != 1.0:
        img = cv2.resize(img, (int(img_w / window_scale), int(img_h / window_scale)))

    # Calculate hog blocks for each desired image channel
    hog_channels = get_hog_features(img, **fvb.hog_param)

    # Number of hog blocks for this image in the xy direction
    nblocks_x = hog_channels[0].shape[1]
    nblocks_y = hog_channels[0].shape[0]

    # The number of blocks we need per window (as required by classifier).
    window_size = fvb.input_img_shape[0]
    cells_per_window_edge = window_size // fvb.hog_param['pixels_per_cell_edge']
    blocks_per_window_edge = \
        cells_per_window_edge - fvb.hog_param['cells_per_block_edge'] + 1

    # Step through all the windows by block increments
    desired_windows = []
    cells_per_window_step = int((1 - window_overlap) * cells_per_window_edge)
    for x_block_step in range((nblocks_x - blocks_per_window_edge) // cells_per_window_step):
        for y_block_step in range((nblocks_y - blocks_per_window_edge) // cells_per_window_step):
            # Get HOG features
            xb_begin, yb_begin = [step * cells_per_window_step for step in (x_block_step, y_block_step)]
            xb_end, yb_end = [begin + blocks_per_window_edge for begin in (xb_begin, yb_begin)]
            hog_features = hog_channels[:, yb_begin:yb_end, xb_begin:xb_end].ravel()

            # Get image patch for this window
            x_px_begin = xb_begin * fvb.hog_param['pixels_per_cell_edge']
            y_px_begin = yb_begin * fvb.hog_param['pixels_per_cell_edge']
            x_px_end, y_px_end = [begin + window_size for begin in (x_px_begin, y_px_begin)]
            window_img = cv2.resize(img[y_px_begin:y_px_end, x_px_begin:x_px_end], fvb.input_img_shape[:2])

            # Get feature vector and classify
            feature_vector = fvb.get_features_single((window_img, hog_features))
            classification = clf.predict(feature_vector)

            if classification == CARS:
                # Transform back image patch coords back to original image space and save as rectangle
                x_rec_left = int(x_px_begin * window_scale)
                y_rec_top = int(y_px_begin * window_scale) + y_range[0]
                draw_size = int(window_size * window_scale)
                desired_windows.append(((x_rec_left, y_rec_top),
                                        (x_rec_left + draw_size, y_rec_top + draw_size)))
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
    fvb = CarFeatureVectorBuilder(feature_scaler=scaler)

    # Test images
    print('Searching for cars in image...')
    for imgf in sorted(glob('./data/test_images/*.jpg')):
        image = mpimg.imread(imgf)
        car_windows = find_cars(image, svc, fvb, y_range=(400, 656), window_scale=1.5, window_overlap=0.75)

        window_img = draw_rectangles(image, car_windows, color=(0, 0, 255), thick=6)
        plt.figure()
        plt.imshow(window_img)
    plt.show()