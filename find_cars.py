#!/usr/bin/env python
"""
Locates cars in video and places bounding boxes around them.

Author: Peter Moran
Created: 9/14/2017
"""
import argparse
from glob import glob
from typing import Tuple, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from feature_extraction import get_hog_features, CarFeatureVectorBuilder

Rectangle = Tuple[Tuple[int, int], Tuple[int, int]]
'''A pair of (x, y) vertices defining a rectangle.'''


def draw_rectangles(img, rectangles: Iterable[Rectangle], color=(0, 0, 255), thick=6):
    """ Draws rectangles on a copy of the given image. """
    imcopy = np.copy(img)
    for bbox in rectangles:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def gen_heatmap(rectangles: Iterable[Rectangle], rectangle_scores, img_shape):
    heatmap = np.zeros(img_shape[:2])
    for rec, score in zip(rectangles, rectangle_scores):
        heatmap[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0]] += score
    return heatmap


def hot_label_regions(score_img, labels, threshold):
    bboxes = []
    for label_id in range(1, labels[1] + 1):
        # Find the location of all pixels assigned to label_id
        pixel_locs = (labels[0] == label_id).nonzero()

        # Find the min/max x, y value of all the pixel_locs for this label
        pixels_y = np.array(pixel_locs[0])
        pixels_x = np.array(pixel_locs[1])
        bbox = ((np.min(pixels_x), np.min(pixels_y)), (np.max(pixels_x), np.max(pixels_y)))

        # Check score at the core is enough
        if score_img[(bbox[0][1] + bbox[1][1]) // 2, (bbox[0][0] + bbox[1][0]) // 2] >= threshold:
            bboxes.append(bbox)

    return bboxes


class CarFinder:
    def __init__(self, classifier, feature_vector_builder, visualization, history=15, thresh_low=0.1, thresh_high=0.3):
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        self.clf = classifier
        self.fvb = feature_vector_builder
        self.viz = visualization
        self.heatmap_history = history
        self.nlast_heatmaps = []

    def window_search_cars(self, img, y_range, window_scale, window_overlap):
        img = np.copy(img)[y_range[0]:y_range[1], :, :]
        img_h, img_w = img.shape[:2]

        # Scale image to search for different size objects
        if window_scale != 1.0:
            img = cv2.resize(img, (int(img_w / window_scale), int(img_h / window_scale)))

        # Calculate hog blocks for each desired image channel
        hog_channels = get_hog_features(img, **self.fvb.hog_param)

        # Number of hog blocks for this image in the xy direction
        nblocks_x = hog_channels[0].shape[1]
        nblocks_y = hog_channels[0].shape[0]

        # The number of blocks we need per window (as required by classifier).
        window_size = self.fvb.input_img_shape[0]
        cells_per_window_edge = window_size // self.fvb.hog_param['pixels_per_cell_edge']
        blocks_per_window_edge = \
            cells_per_window_edge - self.fvb.hog_param['cells_per_block_edge'] + 1

        # Load all images and hog features
        window_imgs = []
        window_hogs = []
        window_positions = []
        cells_per_window_step = int((1 - window_overlap) * cells_per_window_edge)
        for x_block_step in range((nblocks_x - blocks_per_window_edge) // cells_per_window_step):
            for y_block_step in range((nblocks_y - blocks_per_window_edge) // cells_per_window_step):
                # Get HOG features
                xb_begin, yb_begin = [step * cells_per_window_step for step in (x_block_step, y_block_step)]
                xb_end, yb_end = [begin + blocks_per_window_edge for begin in (xb_begin, yb_begin)]
                window_hogs.append(hog_channels[:, yb_begin:yb_end, xb_begin:xb_end].ravel())

                # Get image patch for this window
                x_px_begin = xb_begin * self.fvb.hog_param['pixels_per_cell_edge']
                y_px_begin = yb_begin * self.fvb.hog_param['pixels_per_cell_edge']
                x_px_end, y_px_end = [begin + window_size for begin in (x_px_begin, y_px_begin)]
                window_imgs.append(
                    cv2.resize(img[y_px_begin:y_px_end, x_px_begin:x_px_end], self.fvb.input_img_shape[:2]))

                # Transform back image patch coords back to original image space and save as rectangle
                x_rec_left = int(x_px_begin * window_scale)
                y_rec_top = int(y_px_begin * window_scale) + y_range[0]
                draw_size = int(window_size * window_scale)
                window_positions.append(((x_rec_left, y_rec_top),
                                         (x_rec_left + draw_size, y_rec_top + draw_size)))

        # Get feature vector and classify in a batch
        X = self.fvb.get_features(list(zip(window_imgs, window_hogs)))
        classification_scores = self.clf.decision_function(X)

        # Select for desired windows
        desired_windows = []
        desired_windows_mag = []
        min_score = 0
        for i, score in enumerate(classification_scores):
            if score > min_score:
                desired_windows.append(window_positions[i])
                desired_windows_mag.append(score)
        return desired_windows, desired_windows_mag

    def find_cars(self, img, single=False):
        # Perform search over multiple scales
        searches = [(380, 660, 1.5, 6 / 8)]
        windows, window_scores = [], []
        for ystart, ystop, scale, overlap in searches:
            w, ws = self.window_search_cars(img, (ystart, ystop), scale, overlap)
            windows += w
            window_scores += ws

        # Make heatmap
        current_heatmap = gen_heatmap(windows, window_scores, img.shape[:2])
        if single:
            heatmap = current_heatmap
        else:
            # Add heatmap to history
            self.nlast_heatmaps.append(current_heatmap)
            if len(self.nlast_heatmaps) > self.heatmap_history:
                self.nlast_heatmaps = self.nlast_heatmaps[-self.heatmap_history:]
            # Integrate over history
            heatmap = sum(self.nlast_heatmaps)

        # Remove very weak parts of the heatmap
        min_heat = len(self.nlast_heatmaps) * self.thresh_low
        max_heat = len(self.nlast_heatmaps) * self.thresh_high
        heatmap_thresh = np.copy(heatmap)
        heatmap_thresh[heatmap < min_heat] = 0

        # Label the heatmap and only keep regions that are hot enough at their center
        labels = label(heatmap_thresh)
        bboxes = hot_label_regions(heatmap_thresh, labels, threshold=max_heat)

        # Visualize
        ret_img = np.copy(img)
        for bbox in bboxes:
            ret_img = cv2.rectangle(ret_img, bbox[0], bbox[1], color=(0, 255, 0), thickness=6)
        if self.viz == 'windows':
            heatmap_limtd = np.copy(heatmap)
            heatmap_limtd[heatmap > max_heat] = max_heat
            cv2.normalize(heatmap_limtd, heatmap_limtd, 0, 255, cv2.NORM_MINMAX)
            heatmap_limtd = heatmap_limtd.astype('uint8')
            heatmap_limtd = cv2.merge((heatmap_limtd, np.zeros_like(heatmap_limtd), np.zeros_like(heatmap_limtd)))
            heatmap_overlay = cv2.addWeighted(ret_img, 0.3, heatmap_limtd, 0.7, 0)
            ret_img = draw_rectangles(heatmap_overlay, windows, color=(0, 0, 180), thick=2)

        return ret_img


def main():
    # Load parameters
    parser = argparse.ArgumentParser(description='Locates cars in video and places bounding boxes around them.',
                                     usage='%(prog)s [ -vi & -vo | -img ] [extra_options]')
    parser.add_argument('-vi', '--video_in', type=str, default='./data/test_videos/test_video.mp4',
                        help='Video to find cars in.')
    parser.add_argument('-vo', '--video_out', type=str, help='Where to save video to.')
    parser.add_argument('-img', '--images_in', type=str,
                        help='Search path (glob style) to test images. Cars will be found in images rather than video.')
    parser.add_argument('-clf', '--clf_savefile', type=str, default='./data/trained_classifier.pkl',
                        help="File path to pickled trained classifier made by 'train.py'")
    parser.add_argument('-sc', '--scaler_savefile', type=str, default='./data/Xy_scaler.pkl',
                        help="File path to pickled StandardScalar made by 'train.py'")
    parser.add_argument('-viz', '--visualization', type=str, default='cars',
                        help="'cars' to draw bounding box around cars or 'windows' to show all the detected windows.")
    args = parser.parse_args()
    name, ext = args.video_in.split('/')[-1].rsplit('.', 1)
    args.video_out = './output/{}_{}.{}'.format(name, args.visualization, ext)

    # Set up car finder
    print("Loading classifier from '{}'.".format(args.clf_savefile))
    clf = joblib.load(args.clf_savefile)
    print("Loading scaler from '{}'.".format(args.scaler_savefile))
    scaler = joblib.load(args.scaler_savefile)
    fvb = CarFeatureVectorBuilder(feature_scaler=scaler)
    car_finder = CarFinder(clf, fvb, args.visualization)

    # Find cars in...
    if args.images_in is not None:  # run on images
        print('\nSearching for cars in images...')
        imgs = []
        files = sorted(glob(args.images_in))
        for imgf in files:
            # Find cars
            image = plt.imread(imgf)
            display_img = car_finder.find_cars(image, single=True)
            imgs.append(display_img)

        n_col = 3
        fig, axes = plt.subplots(len(files) // n_col + 1, n_col)
        for ax, f, img in zip(axes.flatten(), files, imgs):
            ax.imshow(img)
            ax.set_title(f)
            ax.axis('off')
        plt.show()

    else:  # run on video
        print("\nFinding cars in '{}',\nthen saving to  '{}'...".format(args.video_in, args.video_out))
        input_video = VideoFileClip(args.video_in)
        output_video = input_video.fl_image(car_finder.find_cars)
        output_video.write_videofile(args.video_out, audio=False)


if __name__ == '__main__': main()
