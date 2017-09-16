import time
from typing import Sequence, TypeVar, Callable, List

import cv2
import numpy as np
from numpy.core.multiarray import ndarray
from skimage.feature import hog
from sklearn.preprocessing import scale, StandardScaler
from tqdm import tqdm


def rgb2any(image, cspace):
    if cspace != 'RGB':
        conversion = getattr(cv2, 'COLOR_RGB2{}'.format(cspace))
        feature_image = cv2.cvtColor(image, conversion)
    else:
        feature_image = np.copy(image)

    return feature_image


def get_hog_features(image, orientations, pixels_per_cell_edge, cells_per_block_edge, cspace='RGB', channels='ALL'):
    # Apply color conversion
    feature_image = rgb2any(image, cspace)

    # Determine image channels to use
    if channels == 'ALL':
        channels = range(feature_image.shape[2])
    else:
        assert isinstance(channels, Sequence[int]), "`channels` must be a sequence of ints or 'ALL'."

    # Collect HOG features
    hog_channels = []  # channel stored separately
    for channel in channels:
        hog_channels.append(hog(feature_image[:, :, channel],
                                orientations=orientations,
                                pixels_per_cell=(pixels_per_cell_edge, pixels_per_cell_edge),
                                cells_per_block=(cells_per_block_edge, cells_per_block_edge),
                                transform_sqrt=True,
                                block_norm='L2-Hys',
                                feature_vector=False))
    return np.array(hog_channels)


def bin_color_spatial(img, size=(32, 32), cspace='RGB'):
    img = rgb2any(img, cspace)
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins=32, bins_range=(0, 256), cspace='RGB', channels='ALL'):
    # Select channels to include
    img = rgb2any(img, cspace)
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
    # TODO: Switch to functional format
    Sample = TypeVar('Sample')
    ProcessedSample = TypeVar('ProcessedSample')

    def __init__(self, normalize_features=True, normalize_samples=False, feature_scaler=None):
        self.preprocessor_funcs = []
        self.preprocessor_types = []
        self.extractor_funcs = []
        self.extractor_func_names = []
        self.normalize_samples = normalize_samples
        self.normalize_features = normalize_features
        self.feature_scaler = feature_scaler

    def add_preprocessor(self, preprocessor_func: Callable[[Sample], ProcessedSample], type):
        self.preprocessor_funcs.append(preprocessor_func)
        self.preprocessor_types.append(type)

    def add_extractor(self, extractor_func: Callable[[ProcessedSample], ndarray], name=None):
        self.extractor_funcs.append(extractor_func)
        if name is None:
            name = 'Extractor {}'.format(len(self.extractor_funcs))
        self.extractor_func_names.append(name)

    def preprocess(self, o):
        try:
            ndx = self.preprocessor_types.index(type(o))
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('There is no preprocessor for type {}'.format(type(o)),)
            raise
        return self.preprocessor_funcs[ndx](o)

    def get_features(self, samples: List[Sample], verbose=0):
        if verbose >= 2:
            print('Extracting features...')
            t0 = time.time()

        # Determine feature vector size
        extractor_return_lens = []
        for extractor_func in self.extractor_funcs:
            example_ret = extractor_func(self.preprocess(samples[0]))
            assert len(example_ret.shape) == 1, 'All functions added by `add_extractor()` must return 1d numpy ' \
                                                'arrays. Did you forget to call `array.ravel()`?'
            extractor_return_lens.append(len(example_ret))
        feature_vec_len = np.sum(extractor_return_lens)

        # Print diagnostics
        if verbose >= 2:
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
            processed_sample = self.preprocess(sample)

            # Add to the feature vector in chunks from each fvb function
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
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler().fit(X)
            X = self.feature_scaler.transform(X)

            # TODO: Select for only important features?

        if verbose >= 2:
            print('Done (after {:.1f} seconds).'.format(time.time() - t0))

        return X

    def get_features_single(self, sample: Sample, verbose=0):
        return self.get_features([sample], verbose=verbose)


class CarFeatureVectorBuilder(FeatureVectorBuilder):
    # TODO: Switch FeatureVectorBuilder to functional format and use here
    def __init__(self, clf_img_shape=(64, 64, 3), normalize_features=True, normalize_samples=False,
                 feature_scaler=None):
        """
        FeatureVectorBuilder initialized for identifying cars vs notcars.

        Used to ensure consistent feature extraction for all usage cases (eg training and classification).
        """
        self.input_img_shape = clf_img_shape
        assert clf_img_shape[0] == clf_img_shape[1], "CarFeatureVectorBuilder requires square image input."
        super().__init__(normalize_features, normalize_samples, feature_scaler)

        # Set up preprocessors
        self.add_preprocessor(lambda file: self.preprocess_file(file), str)  # filename -> (img, hog features)
        self.add_preprocessor(lambda img_and_hog: img_and_hog, tuple)  # passthrough

        # Set up extractors
        self.hog_param = {'orientations': 9, 'pixels_per_cell_edge': 8, 'cells_per_block_edge': 2,
                          'cspace': 'YCrCb', 'channels': 'ALL'}
        self.add_extractor(
            lambda img_and_hog: img_and_hog[1].ravel(), 'HOG extraction')  # pass through hog features
        self.add_extractor(
            lambda img_and_hog: bin_color_spatial(img_and_hog[0], cspace='YCrCb'), 'Spatial binning')  # bin the img
        self.add_extractor(
            lambda img_and_hog: color_hist(img_and_hog[0], cspace='YCrCb'), 'Color histogram')  # histogram the img

    def preprocess_file(self, file):
        img = cv2.imread(file)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        hog = get_hog_features(img, **self.hog_param).ravel()
        return img, hog

    def preprocess(self, o):
        img, hog = super().preprocess(o)
        assert img.dtype == 'uint8', 'CarFeatureVectorBuilder is initialized uint8 images, not {}'.format(img.dtype)
        assert img.shape == self.input_img_shape, 'CarFeatureVectorBuilder is initialized for images of shape' \
                                                  ' {} not {}'.format(self.input_img_shape, img.shape)
        # Normalize lighting
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
        cv2.normalize(l, l, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        # TODO: Pre-convert to a different default image space?
        return img, hog
