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
import numpy as np
from tqdm import tqdm
from numpy.core.multiarray import ndarray
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle


def hog_features(image, n_orient=9, pix_per_cell=8, cell_per_block=2, cspace='BGR', hog_channels='ALL'):
    # Apply color conversion
    if cspace != 'BGR':
        conversion = getattr(cv2, 'COLOR_BGR2{}'.format(cspace))
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


def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins=32, bins_range=(0, 256), channels='ALL'):
    # Select channels to include
    if channels == 'ALL':
        channels = range(img.shape[2])
    else:
        assert isinstance(channels, Sequence[int]), "`channels` must be a sequence of ints or 'ALL'."

    # Bin each channel and combine
    histograms = []
    for channel in channels:
        histogram, bin_edges = np.histogram(img[:, :, channel], bins=nbins, range=bins_range)
        histograms.append(histogram)
    return np.concatenate(histograms)


class FeatureVectorBuilder:
    Sample = TypeVar('Sample')
    ProcessedSample = TypeVar('ProcessedSample')

    def __init__(self, preprocessor_func: Callable[[Sample], ProcessedSample]):
        self._preprocessor_func = preprocessor_func
        self._extractor_funcs = []

    def add_extractor(self, extractor_func: Callable[[ProcessedSample], ndarray]):
        self._extractor_funcs.append(extractor_func)

    def get_features(self, samples: Sample, verbose=False):
        if verbose:
            print('\nExtracting features...')
            t1 = time.time()

        # Determine feature vector size
        extractor_return_len = []
        for extractor_func in self._extractor_funcs:
            extractor_return_len.append(len(extractor_func(self._preprocessor_func(samples[0]))))
        feature_vec_len = np.sum(extractor_return_len)

        # Find the feature vector for every sample
        feature_vectors = np.zeros(shape=(len(samples), feature_vec_len))
        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            processed_sample = self._preprocessor_func(sample)

            # Add to the feature vector in chunks from each extractor function
            start = 0
            for j, extractor_func in enumerate(self._extractor_funcs):
                stop = start + extractor_return_len[j]
                feature_vectors[i, start:stop] = extractor_func(processed_sample)  # ensure features are flattened
                start = stop

        # Normalize over features (column wise).
        X_normalized = StandardScaler().fit(feature_vectors).transform(feature_vectors)

        if verbose:
            print('Done (after {:.1f} seconds).'.format(time.time() - t1))
            print('Feature vector length:', len(X_normalized[0]))

        return X_normalized


if __name__ == '__main__':
    # Divide up into files_car and files_notcars
    files_car = glob.glob('./data/vehicles/*/*.png')
    files_notcars = glob.glob('./data/non-vehicles/*/*.png')

    print('Total number of car files:', len(files_car))
    print('Total number of notcar files:', len(files_notcars))

    # Reduce the sample size to speed things up
    sample_size = 500
    print('Using {} samples each, {} samples total.'.format(sample_size, sample_size * 2))
    files_car = shuffle(files_car)[:sample_size]
    files_notcars = shuffle(files_notcars)[:sample_size]

    # Define feature extractor
    feature_builder = FeatureVectorBuilder(preprocessor_func=lambda file: cv2.imread(file))
    feature_builder.add_extractor(lambda img: hog_features(img, cspace='HSV', cell_per_block=3))
    feature_builder.add_extractor(bin_spatial)
    feature_builder.add_extractor(color_hist)

    # Extract features
    all_files = files_car + files_notcars
    X = feature_builder.get_features(all_files, verbose=True)
    y = np.hstack((np.ones(len(files_car)), np.zeros(len(files_notcars))))

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Grid search parameters
    param_grid = [
        {'C': [5, 10, 15, 20],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']},
    ]

    # Perform grid search
    print('\nPerforming grid search with SCV...')
    svc = SVC(cache_size=1000)
    clf = GridSearchCV(svc, param_grid, n_jobs=32, verbose=1)
    t0 = time.time()
    clf.fit(X_train, y_train)
    print('Done after {:.2f} seconds.'.format(time.time() - t0, 2))

    # Print stats
    print('\nBest SVC score: {:0.3f}'.format(clf.best_score_))
    print('Best SVC parameters set: {}'.format(clf.best_params_))
    print('\nGrid scores on development set:\n')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('{:0.3f} (+/-{:0.03f}) for {!r}'.format(mean, std * 2, params))