#!/usr/bin/env python
"""
Trains a classifier to recognise car vs not-car images.

Author: Peter Moran
Created: 9/12/2017
"""
import glob
import sys
import time
from typing import TypeVar, Callable, Sequence

import cv2
import numpy as np
from numpy.core.multiarray import ndarray
from scipy.stats import expon
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm


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


def bin_color_spatial(img, size=(32, 32)):
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

    def __init__(self, preprocessor_func: Callable[[Sample], ProcessedSample],
                 normalize_features=True, normalize_samples=False):
        self.preprocessor_func = preprocessor_func
        self.extractor_funcs = []
        self.extractor_func_names = []
        self.normalize_samples = normalize_samples
        self.normalize_features = normalize_features

    def add_extractor(self, extractor_func: Callable[[ProcessedSample], ndarray], name=None):
        self.extractor_funcs.append(extractor_func)
        if name is None:
            name = 'Extractor {}'.format(len(self.extractor_funcs))
        self.extractor_func_names.append(name)

    def get_features(self, samples: Sample, verbose=False):
        if verbose:
            print('\nExtracting features...')
            t0 = time.time()

        # Determine feature vector size
        extractor_return_lens = []
        for extractor_func in self.extractor_funcs:
            example_ret = extractor_func(self.preprocessor_func(samples[0]))
            assert len(example_ret.shape) == 1, 'All functions added by `add_extractor()` must return 1d numpy ' \
                                                'arrays. Did you forget to call `array.ravel()`?'
            extractor_return_lens.append(len(example_ret))
        feature_vec_len = np.sum(extractor_return_lens)

        if verbose:
            print('Feature vector length:', feature_vec_len)
            longest_name = max([len(name) for name in self.extractor_func_names])
            for i in range(len(self.extractor_funcs)):
                ret_len = extractor_return_lens[i]
                print("{:<{fill}} contributes {:>5} features ({:>5.1f}%) to each feature vector.".format(
                    self.extractor_func_names[i], ret_len, (ret_len / feature_vec_len) * 100, fill=longest_name))
                time.sleep(1e-2)  # give print time to finish before progress bar starts

        # Find the feature vector for every sample
        X = np.zeros(shape=(len(samples), feature_vec_len))  # all feature vectors
        for i, sample in tqdm(enumerate(samples), total=len(samples), disable=not verbose):
            processed_sample = self.preprocessor_func(sample)

            # Add to the feature vector in chunks from each extractor function
            start = 0
            for j, extractor_func in enumerate(self.extractor_funcs):
                features = extractor_func(processed_sample)

                if self.normalize_samples:
                    # Normalize over sample. We do this separately for features returned from different functions.
                    features = features.astype('float64', copy=False)  # change type if needed
                    scale(features.reshape(-1, 1), axis=0, copy=False)

                # Fill in this function's segment of the feature vector.
                stop = start + extractor_return_lens[j]
                X[i, start:stop] = features
                start = stop

        if self.normalize_features:
            # Normalize over features (column wise).
            scale(X, copy=False)

        if verbose:
            print('Done (after {:.1f} seconds).'.format(time.time() - t0))

        return X


class CarFeatureVectorBuilder(FeatureVectorBuilder):
    def __init__(self):
        """
        FeatureVectorBuilder initialized for identifying cars vs notcars.

        Used to ensure consistent feature extraction for all usage cases (eg training and classification).
        """
        super().__init__(lambda file: cv2.imread(file))
        self.add_extractor(lambda img: hog_features(img, cspace='HSV', cell_per_block=3), 'HOG extraction')
        self.add_extractor(bin_color_spatial, 'Spatial binning')
        self.add_extractor(color_hist, 'Color histogram')


if __name__ == '__main__':
    # Read arguments
    argc = len(sys.argv)
    arg1 = sys.argv[1] if argc > 1 else '1000'
    train_iters = int(sys.argv[2]) if argc > 2 else 10
    train_jobs = int(sys.argv[3]) if argc > 3 else 4
    clf_savefile = sys.argv[4] if argc > 4 else 'trained_classifier.pkl'
    Xy_savefile = sys.argv[5] if argc > 5 else None

    if arg1.isdigit():
        # Build X and y
        sample_size = int(arg1)

        # Divide up into files_car and files_notcars
        files_car = glob.glob('./data/vehicles/*/*.png')
        files_notcars = glob.glob('./data/non-vehicles/*/*.png')

        print('Total number of car files:', len(files_car))
        print('Total number of notcar files:', len(files_notcars))

        # Reduce the sample size to speed things up
        print('Using {} samples each, {} samples total.'.format(sample_size, sample_size * 2))
        files_car = shuffle(files_car)[:sample_size]
        files_notcars = shuffle(files_notcars)[:sample_size]

        # Define feature extractor
        feature_builder = CarFeatureVectorBuilder()

        # Extract features
        X_files = files_car + files_notcars
        X = feature_builder.get_features(X_files, verbose=True)
        y = np.hstack((np.ones(len(files_car)), np.zeros(len(files_notcars))))
        Xy_savefile = 'Xy.pkl' if Xy_savefile is None else Xy_savefile
        print("Saving features to '{}'.".format(Xy_savefile))
        joblib.dump((X, y), Xy_savefile)

    else:
        # Read X and y from file
        print("Loading `X` any `y` from '{}'.".format(arg1))
        X, y = joblib.load(arg1)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Grid search parameters
    param_dist = {'C': expon(scale=20),
                  'gamma': expon(scale=.0005),
                  'kernel': ['rbf']}

    # Perform grid search
    print('\nPerforming grid search with SCV...')
    svc = SVC(cache_size=1000)
    random_search = RandomizedSearchCV(svc, param_dist, n_jobs=train_jobs, n_iter=train_iters, verbose=1)
    t0 = time.time()
    random_search.fit(X_train, y_train)
    print('Done after {:.2f} seconds.'.format(time.time() - t0, 2))

    # Print stats
    print('\nGrid scores on development set:')
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
        print('\t{:0.3f} (+/-{:0.03f}) for {!r}'.format(mean, std * 2, params))
    print('\nBest SVC score: {:0.3f}'.format(random_search.best_score_))
    print('Best SVC parameters set: {}'.format(random_search.best_params_))

    # Save the feature extractor and best model
    print('\nSaving best classifier to "{}".'.format(clf_savefile))
    joblib.dump(random_search.best_estimator_, clf_savefile)
