#!/usr/bin/env python
"""
Trains a classifier to recognise car vs not-car images.

See README.md or run `train.py -h` for usage.

Author: Peter Moran
Created: 9/12/2017
"""
import argparse
import time
from glob import glob

import numpy as np
from scipy.stats import expon
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from feature_extraction import CarFeatureVectorBuilder

NOTCARS = 0
CARS = 1


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description='Train classifier for cars vs not-cars.',
                                     usage='%(prog)s [(-cf & -ncf)? -sz | -xylf ] [additional options]')
    parser.add_argument('-sz', '--sample_size', type=int, help='Number of samples to use from both -cf and -ncf.')
    parser.add_argument('-cf', '--car_pickle_file', type=str,
                        help='File path to .pkl containing a list of image files to use as cars.')
    parser.add_argument('-ncf', '--notcar_pickle_file', type=str,
                        help='File path to .pkl containing a list of image files to use as notcars.')
    parser.add_argument('-xylf', '--xy_loadfile', type=str,
                        help='File path to .pkl containing existing feature vectors and labels as tuple (X, y).')
    parser.add_argument('-ti', '--train_iters', type=int, default=10,
                        help='Number of iterations to use for the SVM parameter search.')
    parser.add_argument('-tj', '--train_jobs', type=int, default=4,
                        help='Number of processes to use for the SVM parameter search.')
    parser.add_argument('-clf', '--clf_savefile', type=str, default='./data/trained_classifier.pkl',
                        help='Location to save the classifier.')
    parser.add_argument('-xysf', '--xy_savefile', type=str, default='./data/Xy.pkl',
                        help='Location to save the extracted feature vectors and labels used in training as a .pkl.')
    parser.add_argument('-sr', '--save_remainder', type=bool, default=False,
                        help="Set 'True' to also extract the unused features and save them to a .pkl file.")
    args = parser.parse_args()
    if args.save_remainder and args.sz is None:
        parser.error("--save_remainder requires using --sample_size.")
    if (args.car_pickle_file is not None) ^ (args.notcar_pickle_file is not None):
        parser.error("-cf and -ncf must be passed together.")
    if (args.car_pickle_file is not None) and (args.notcar_pickle_file is not None) and args.sample_size is None:
        parser.error("-cf and -ncf require -sz to be passed.")

    # Get image features
    if args.xy_loadfile is not None:  # Read X and y from file
        print("Loading `X` any `y` from '{}'.".format(args.xy_loadfile))
        X, y = joblib.load(args.xy_loadfile)
    else:  # Extract X and y from images
        if args.car_pickle_file is not None and args.notcar_pickle_file is not None:
            # Read car_files and notcar_files from pickled list
            car_files = joblib.load(args.car_pickle_file)
            notcar_files = joblib.load(args.notcar_pickle_file)
        else:
            # Automatically load all car_files and notcar_files
            car_files = glob('./data/vehicles/*/*.png')
            notcar_files = glob('./data/non-vehicles/*/*.png')

        # Shuffle
        car_files = shuffle(car_files)
        notcar_files = shuffle(notcar_files)

        print('Total number of car files:', len(car_files))
        print('Total number of notcar files:', len(notcar_files))

        # Reduce the sample size to speed things up
        n_smaller_class = min(len(car_files), len(notcar_files))
        assert args.sample_size <= n_smaller_class, 'Training on unbalanced sets not supported.'
        print('Using {} samples each, {} samples total.\n'.format(args.sample_size, args.sample_size * 2))

        # Define feature fvb
        fvb = CarFeatureVectorBuilder()

        # Extract features
        X_files = car_files[:args.sample_size] + notcar_files[:args.sample_size]
        X = fvb.get_features(X_files, verbose=2)
        y = np.hstack((np.ones(args.sample_size), np.zeros(args.sample_size)))

        # Normalize across features
        fvb.feature_scaler = StandardScaler().fit(X)
        X = fvb.feature_scaler.transform(X)

        # Save features
        xy_savepath, ext = args.xy_savefile.rsplit('.', 1)
        print("Saving features to '{}'.".format(args.xy_savefile))
        joblib.dump((X, y), args.xy_savefile, compress=3)
        scaler_save_file = '{}_scaler.{}'.format(xy_savepath, ext)
        print("Saving `StandardScaler` for this feature set to '{}'.".format(scaler_save_file))
        joblib.dump(fvb.feature_scaler, scaler_save_file)

        if args.save_remainder:
            n_remainder = n_smaller_class - args.sample_size
            Xy_unused_save_file = '{}_test.{}'.format(xy_savepath, ext)
            print('\nExtracting an additional {} samples from each set ({} in total) that were unused. This can act as'
                  'an additional test set.'.format(n_remainder, n_remainder * 2))

            # Build unused features and save
            X_unused_files = \
                car_files[args.sample_size: n_smaller_class] + notcar_files[args.sample_size: n_smaller_class]
            X_unused = fvb.get_features(X_unused_files, verbose=1)
            y_unused = np.hstack((np.ones(n_remainder), np.zeros(n_remainder)))
            print("Done. Saving to '{}'.".format(Xy_unused_save_file))
            joblib.dump((X_unused, y_unused), Xy_unused_save_file)

    # Split into train/validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)

    # Grid search parameters
    param_dist = {'C': expon(scale=100),
                  'gamma': expon(scale=.001),
                  'kernel': ['rbf']}

    # Perform grid search
    print('\nPerforming grid search with SCV on {} threads...'.format(args.train_jobs))
    svc = SVC(cache_size=1000)
    random_search = RandomizedSearchCV(svc, param_dist, n_jobs=args.train_jobs, n_iter=args.train_iters, verbose=2)
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
    print('Number of support vectors (impacts prediction time): {}'.format(random_search.best_estimator_.n_support_))

    # Save the feature fvb and best model
    print('\nSaving best classifier to "{}".'.format(args.clf_savefile))
    joblib.dump(random_search.best_estimator_, args.clf_savefile)


if __name__ == '__main__': main()
