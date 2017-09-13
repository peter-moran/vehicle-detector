#!/usr/bin/env python
"""
Trains a classifier to recognise car vs not-car images.

Author: Peter Moran
Created: 9/12/2017
"""
import glob
import time
from typing import TypeVar, Callable, Sequence

import cv2
import matplotlib.image as mpimg
import numpy as np
from numpy.core.multiarray import ndarray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


class FeatureVectorBuilder:
    Sample = TypeVar('Sample')
    ProcessedSample = TypeVar('ProcessedSample')

    def __init__(self, preprocessor_func: Callable[[Sample], ProcessedSample]):
        self._preprocessor_func = preprocessor_func
        self._extractor_funcs = []

    def add_extractor(self, extractor_func: Callable[[ProcessedSample], ndarray]):
        self._extractor_funcs.append(extractor_func)

    def get_features(self, samples: Sample):
        # Determine feature vector size
        extractor_return_len = []
        for extractor_func in self._extractor_funcs:
            extractor_return_len.append(len(extractor_func(self._preprocessor_func(samples[0]))))
        feature_vec_len = np.sum(extractor_return_len)

        # Find the feature vector for every sample
        feature_vectors = np.zeros(shape=(len(samples), feature_vec_len))
        for i, sample in enumerate(samples):
            processed_sample = self._preprocessor_func(sample)

            # Add to the feature vector in chunks from each extractor function
            start = 0
            for j, extractor_func in enumerate(self._extractor_funcs):
                stop = start + extractor_return_len[j]
                feature_vectors[i, start:stop] = extractor_func(processed_sample)  # ensure features are flattened
                start = stop

        # Normalize over features (column wise).
        X_normalized = StandardScaler().fit(feature_vectors).transform(feature_vectors)
        return X_normalized


def extract_hog_features(image: ndarray, n_orient=9, pix_per_cell=8, cell_per_block=2, cspace='RGB',
                         hog_channels='ALL'):
    # Apply color conversion
    if cspace != 'RGB':
        conversion = getattr(cv2, 'COLOR_RGB2{}'.format(cspace))
        feature_image = cv2.cvtColor(image, conversion)
    else:
        feature_image = np.copy(image)

    # Determine image channels to use
    if hog_channels == 'ALL':
        hog_channels = range(feature_image.shape[2])
    else:
        assert isinstance(hog_channels, Sequence[int]), "`hog_channels` must be a sequence of ints or 'ALL'."

    # Collect HOG features
    hog_features = []
    for channel in hog_channels:
        hog_features.append(hog(feature_image[:, :, channel],
                                orientations=n_orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,
                                block_norm='L2-Hys',
                                feature_vector=True))
    return np.ravel(hog_features)


if __name__ == '__main__':
    # Divide up into files_car and files_notcars
    files_car = glob.glob('./data/vehicles/*/*.png')
    files_notcars = glob.glob('./data/non-vehicles/*/*.png')

    print('Total number of car all_files:', len(files_car))
    print('Total number of notcar all_files:', len(files_notcars))

    # Reduce the sample size to speed things up
    sample_size = 500
    print('Using {} samples each, {} total.'.format(sample_size, sample_size * 2))
    files_car = shuffle(files_car)[:sample_size]
    files_notcars = shuffle(files_notcars)[:sample_size]

    # Define feature extractor
    feature_builder = FeatureVectorBuilder(preprocessor_func=lambda file: mpimg.imread(file))
    feature_builder.add_extractor(lambda img: extract_hog_features(img, cspace='HSV', cell_per_block=3))

    # Extract features
    all_files = files_car + files_notcars
    print('\nExtracting features...')
    t1 = time.time()
    X = feature_builder.get_features(all_files)
    y = np.hstack((np.ones(len(files_car)), np.zeros(len(files_notcars))))
    print('Done (after {:.1f} seconds).'.format(time.time() - t1))
    print('Feature vector length:', len(X[0]))

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Use a linear SVC
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('\nIt took', round(t2 - t, 2), 'seconds to train.')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))