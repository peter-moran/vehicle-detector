#!/usr/bin/env python
"""
Contains all functions needed by this project to extract features from images.

Author: Peter Moran
Created: 9/16/2017
"""

import time
from typing import Sequence, Tuple

import cv2
import numpy as np
from numpy.core.multiarray import ndarray
from skimage.feature import hog
from sklearn.preprocessing import scale
from tqdm import tqdm


def cspace_transform(img, c_from, c_to) -> ndarray:
    """ Transforms any image from one color space to another, if possible. Only makes a copy if new color space. """
    if c_to != c_from:
        conversion = getattr(cv2, 'COLOR_{}2{}'.format(c_from, c_to))
        feature_image = cv2.cvtColor(img, conversion)
    else:
        feature_image = img
    return feature_image


def get_hog_features(img, orientations, pixels_per_cell_edge, cells_per_block_edge, c_from, c_to, channels='ALL'):
    """
    Performs standard HOG feature extraction, but on multiple color channels and with optional color space conversion.

    Parameters `c_from` and `c_to` allow for color space conversion before running.
    Returns a numpy array with each entry containing the HOG map for each color channel.
    """
    # Apply color conversion
    img = cspace_transform(img, c_from, c_to)

    # Determine img channels to use
    if channels == 'ALL':
        channels = range(img.shape[2])
    else:
        assert isinstance(channels, Sequence[int]), "`channels` must be a sequence of ints or 'ALL'."

    # Collect HOG features
    hog_channels = []  # channel stored separately
    for channel in channels:
        hog_channels.append(hog(img[:, :, channel],
                                orientations=orientations,
                                pixels_per_cell=(pixels_per_cell_edge, pixels_per_cell_edge),
                                cells_per_block=(cells_per_block_edge, cells_per_block_edge),
                                transform_sqrt=True,
                                block_norm='L2-Hys',
                                feature_vector=False))
    return np.array(hog_channels)


def bin_color_spatial(img, c_from, c_to, size=(32, 32)):
    """ Returns a feature vector containing the color intensity of the image in spatial order.

    Parameters `c_from` and `c_to` allow for color space conversion before running.
    Changing `size` allows for spatial binning via resizing the image.
    """
    img = cspace_transform(img, c_from, c_to)
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, c_from, c_to, nbins=32, bins_range=(0, 256), channels='ALL'):
    """ Returns the concatenation of color intensity histograms for each specified color channel.

    Parameters `c_from` and `c_to` allow for color space conversion before running.
    """
    # Select channels to include
    img = cspace_transform(img, c_from, c_to)
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


def generate_feature_vectors(samples, extractor_funcs, extractor_func_names, preprocessor_func=None,
                             feature_scaler=None, normalize_samples=False, verbose=0):
    """
    Runs feature extraction pipeline as defined by given preprocessor_func and extractor_funcs. Returns data array X.

    For each sample:
        1. Pass the sample through the `preprocessor_func`
        2. Pass the result of the preprocessor through each function in `extractor_funcs` and concatenate the results
            * Optionally, normalize the samples before concatenation.
        3. Optionall, normalize across features with the given StandardScaler

    In addition, this routine returns various diagnostic information if `verbose` is set greater than zero.

    :type verbose: int
    :type preprocessor_func: Callable[[T], V]
    :type feature_scaler: sklearn.preprocessing.StandardScaler
    :type extractor_func_names: List[str]
    :type extractor_funcs: List[Callable[[V], ndarray]]
    :type samples: List[T]
    """
    if preprocessor_func is None:
        preprocessor_func = lambda sample: sample  # just pass the sample through

    if verbose >= 2:
        print('Extracting features...')
        t0 = time.time()

    # Determine feature vector size
    extractor_return_lens = []
    for extractor_func in extractor_funcs:
        example_ret = extractor_func(preprocessor_func(samples[0]))
        assert len(example_ret.shape) == 1, 'All functions added by `add_extractor()` must return 1d numpy ' \
                                            'arrays. Did you forget to call `array.ravel()`?'
        extractor_return_lens.append(len(example_ret))
    feature_vec_len = np.sum(extractor_return_lens)

    # Print diagnostics
    if verbose >= 2:
        print('Feature vector length:', feature_vec_len)
        longest_name = max([len(name) for name in extractor_func_names])
        for i in range(len(extractor_funcs)):
            ret_len = extractor_return_lens[i]
            print("{:<{fill}} contributes {:>5} features ({:>5.1f}%) to each feature vector.".format(
                extractor_func_names[i], ret_len, (ret_len / feature_vec_len) * 100, fill=longest_name))
            time.sleep(1e-2)  # give print time to finish before progress bar starts

    # Find the feature vector for every sample
    X = np.zeros(shape=(len(samples), feature_vec_len))  # all feature vectors
    for i, sample in tqdm(enumerate(samples), total=len(samples), disable=(verbose == 0)):
        processed_sample = preprocessor_func(sample)

        # Add to the feature vector in chunks from each fvb function
        start = 0
        for j, extractor_func in enumerate(extractor_funcs):
            features = extractor_func(processed_sample)

            if normalize_samples:
                # Normalize over sample. We do this separately for features returned from different functions.
                features = features.astype('float64', copy=False)  # change type if needed
                scale(features.reshape(-1, 1), axis=0, copy=False)

            # Fill in this function'sample segment of the feature vector.
            stop = start + extractor_return_lens[j]
            X[i, start:stop] = features
            start = stop

    # Normalize over features? (column wise)
    if feature_scaler is not None:
        X = feature_scaler.transform(X)

    if verbose >= 2:
        print('Done (after {:.1f} seconds).'.format(time.time() - t0))

    return X


class CarFeatureVectorBuilder:
    def __init__(self, clf_img_shape=(64, 64, 3), feature_scaler=None):
        """
        Central interface to tuning feature extraction specifically for cars vs not-cars.

        The self.feature_scaler member allows for passing in a sklearn StandardScaler obtained during training so that
        all future calls to get_features() return feature vectors that are normalized in the same way.

        CarFeatureVectorBuilder is capable of running on both file names and images with precomputed hog features. See
        self.get_features() for more info.

        :param clf_img_shape: Image shape expected by the classifier.
        :param feature_scaler: A sklearn StandardScaler to use in feature extraction.
        """
        # Read in and check parameters
        self.feature_scaler = feature_scaler
        self.input_img_shape = clf_img_shape
        assert clf_img_shape[0] == clf_img_shape[1], "CarFeatureVectorBuilder requires square image input."

        # Global feature extraction settings.
        self.cspace_def = 'YCrCb'  # default color space (to convert to during preprocessing)
        self.hog_param = {'orientations': 9, 'pixels_per_cell_edge': 8, 'cells_per_block_edge': 2, 'c_from': 'RGB',
                          'c_to': 'YCrCb', 'channels': 'ALL'}

        # Set up extractors. All will expect input to be a tuple (image_patch, hog_features).
        self.feat_extract_funcs = [lambda sample: sample[1].ravel(),
                                   lambda sample: bin_color_spatial(sample[0], c_from=self.cspace_def, c_to='YCrCb'),
                                   lambda sample: color_hist(sample[0], c_from=self.cspace_def, c_to='YCrCb')]
        self.extractor_func_names = ['HOG features', 'Spatial histogram', ' Color histogram']

    def get_features(self, samples, verbose=0):
        """
        Returns a data array X containing the feature vector for each given sample.

        `samples` can be either:
            * a list of file names, in which case all features will be automatically calculated, or
            * a tuple containing an image and pre-computed hog features. Image should be in RGB color space.
        """
        # Set up preprocessor.
        if isinstance(samples[0], str):
            preprocess = self._preprocess_files
        elif isinstance(samples[0], tuple):
            preprocess = self._preprocess_img_hog
        else:
            raise Exception('Sample not formatted correctly.')

        # Extract features
        X = generate_feature_vectors(samples, self.feat_extract_funcs, self.extractor_func_names, preprocess,
                                     self.feature_scaler, verbose=verbose)
        return X

    def _preprocess_img_hog(self, sample: Tuple[ndarray, ndarray]):
        """ Preprocessor used by `self.get_features()` when samples are (img, hog) pairs. """
        img, hog = sample
        assert img.dtype == 'uint8', 'CarFeatureVectorBuilder is initialized uint8 images, not {}'.format(img.dtype)
        assert img.shape == self.input_img_shape, 'CarFeatureVectorBuilder is initialized for images of shape' \
                                                  ' {} not {}'.format(self.input_img_shape, img.shape)
        # Normalize lighting
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
        cv2.normalize(l, l, 0, 255, cv2.NORM_MINMAX)
        img = cspace_transform(cv2.merge((l, a, b)), 'LAB', 'RGB')
        img = cspace_transform(img, 'RGB', self.cspace_def)

        return img, hog

    def _preprocess_files(self, sample: str):
        """ Preprocessor used by `self.get_features()` when samples are files. """
        img = cv2.imread(sample)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        hog = get_hog_features(img, **self.hog_param).ravel()
        return self._preprocess_img_hog((img, hog))
