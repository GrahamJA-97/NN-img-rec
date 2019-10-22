#    name: KerasTestDigitModel.py
#  author: molloykp (Oct 2019)
# purpose: Loads a Kera model designed to accept 28 by 28 images and
#          tests the model on some data.  Calls pa2pre
#

#   modifications:
###
#   molloykp Oct 2019
#     Dynamically import preprocessing routines from command
#     line parameter.
##################################

import numpy as np
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import importlib

from keras.models import load_model
import argparse

#from pa2pre import processTestData

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Test kera model on MNIST data')

    parser.add_argument('--modelFile', action='store',
                        dest='modelFile', default="", required=True,
                        help='the h5 file to load to test')
    parser.add_argument('--test_X_file', action='store',
                        dest='test_X_file', default="", required=True,
                        help='data')

    parser.add_argument('--test_y_file', action='store',
                       dest='test_y_file', default="", required=True,
                       help='file to store matrix with noise')

    parser.add_argument('--preFileName', action='store',
                       dest='preFileName', default="", required=True,
                       help='file name with preprocessing code')

    parser.add_argument('-verbose', action='store_true',
                        dest='verbose', default=False, required=False,
                        help='Verbose flag')
    return parser.parse_args()


def main():
    parms = parseArguments()

    # remove .py from filename (if supplied)
    dotOffset = parms.preFileName.find('.')
    if dotOffset != -1:
        parms.preFileName = parms.preFileName[0:dotOffset]

    print('reading in module:', parms.preFileName)

    # load whatever module contains preprocessin and place
    # it in the 'pre' namespace

    pre = importlib.import_module(parms.preFileName)

    X_test = np.load(parms.test_X_file)
    y_test = np.load(parms.test_y_file)

    (X_test, y_test) = pre.processTestData(X_test,y_test)

    print('reading in model file:', parms.modelFile)

    model = load_model(parms.modelFile)
    model.summary()

    print (X_test.shape[0], ' test samples')

    score = model.evaluate(X_test,y_test,verbose=parms.verbose)

    print('Test score:', score[0], ' Test accuracy:', score[1])
    # used for Autolab to detect accuracy.  Works since valid
    # values are only between 0 and 100 (and return codes
    # can be between -128 and 127).

    sys.exit(int(score[1]*100))


if __name__ == '__main__':
    main()
