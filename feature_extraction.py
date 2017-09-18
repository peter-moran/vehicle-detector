import time
from typing import Sequence, Tuple, TypeVar

import cv2
import numpy as np
from numpy.core.multiarray import ndarray
from skimage.feature import hog
from sklearn.preprocessing import scale
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


T = TypeVar('T')
V = TypeVar('V')


def generate_feature_vectors(samples, extractor_funcs, extractor_func_names, preprocessor_func=None,
                             feature_scaler=None, normalize_samples=False, verbose=0):
    """
    :type verbose: int
    :type preprocessor_func: Callable[[T], V]
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

    if feature_scaler is not None:
        # Normalize over features (column wise).
        X = feature_scaler.transform(X)

    if verbose >= 2:
        print('Done (after {:.1f} seconds).'.format(time.time() - t0))

    return X


class CarFeatureVectorBuilder:
    def __init__(self, clf_img_shape=(64, 64, 3), feature_scaler=None):
        # Read in and check parameters
        self.feature_scaler = feature_scaler
        self.input_img_shape = clf_img_shape
        assert clf_img_shape[0] == clf_img_shape[1], "CarFeatureVectorBuilder requires square image input."

        # Global feature extraction settings.
        self.hog_param = {'orientations': 9, 'pixels_per_cell_edge': 8, 'cells_per_block_edge': 2, 'cspace': 'YCrCb',
                          'channels': 'ALL'}

    def get_features(self, samples, verbose=0):
        # Set up preprocessor.
        if isinstance(samples[0], str):
            preprocess = self._preprocess_files
        elif isinstance(samples[0], tuple):
            preprocess = self._preprocess_img_hog
        else:
            raise Exception('Sample not formatted correctly.')

        # Set up extractors. All will expect input to be a tuple (image_patch, hog_features).
        extractor_func_names = ['HOG features', 'Spatial histogram', ' Color histogram']
        feat_extract_funcs = [lambda sample: sample[1].ravel(),
                              lambda sample: bin_color_spatial(sample[0], cspace='YCrCb'),
                              lambda sample: color_hist(sample[0], cspace='YCrCb')]

        # Extract features
        X = generate_feature_vectors(samples, feat_extract_funcs, extractor_func_names, preprocess,
                                     self.feature_scaler, verbose=verbose)
        return X

    def _preprocess_files(self, sample: str):
        img = cv2.imread(sample)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        hog = get_hog_features(img, **self.hog_param).ravel()
        return self._preprocess_img_hog((img, hog))

    def _preprocess_img_hog(self, sample: Tuple[ndarray, ndarray]):
        img, hog = sample
        assert img.dtype == 'uint8', 'CarFeatureVectorBuilder is initialized uint8 images, not {}'.format(img.dtype)
        assert img.shape == self.input_img_shape, 'CarFeatureVectorBuilder is initialized for images of shape' \
                                                  ' {} not {}'.format(self.input_img_shape, img.shape)
        # Normalize lighting
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
        cv2.normalize(l, l, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        # TODO: Pre-convert to a different default image space?
        return img, hog
