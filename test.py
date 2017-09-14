#!/usr/bin/env python
"""
Simple script for testing a saved classifier on a saved dataset.

Author: Peter Moran
Created: 9/13/2017
"""
import sys
from time import time

from sklearn.externals import joblib

argc = len(sys.argv)
Xy_savefile = sys.argv[1] if argc > 1 else './data/Xy_test.pkl'
clf_savefile = sys.argv[2] if argc > 2 else './data/trained_classifier.pkl'

print("Loading `X` any `y` from '{}'.".format(Xy_savefile))
X, y = joblib.load(Xy_savefile)
print("Loading classifier from '{}'.".format(clf_savefile))
clf = joblib.load(clf_savefile)

print('Testing {} samples...'.format(X.shape[0]))
t0 = time()
score = clf.score(X, y)
print('Done (after {:.1f} seconds).'.format(time() - t0))
print('Mean accuracy: {}'.format(score))
