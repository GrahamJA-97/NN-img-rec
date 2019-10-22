#    name: pa2pre.py
# purpose: Student's add code to pre-processing of the data

# Recall that any pre-processing you do on your training
# data, you must also do on any future data you want to
# predict.  This file allows you to perform any
# pre-processing you need on my undisclosed test data

import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing

NB_CLASSES = 10


def processTestData(X, y):

    # X pre-processing goes here -- students optionally complete
    # reshapes and normalizes our test data.
    n = X.shape[0] # need to adjust the reshape to account for different sized datasets (n, 28, 28)

    X = X.reshape(10000, 784)
    min_max_scaler = preprocessing.MinMaxScaler()
    Xscaled = min_max_scaler.fit_transform(X)

    # y pre-processing goes here.  y_test becomes a ohe
    y_ohe = np_utils.to_categorical (y, NB_CLASSES)

    return Xscaled, y_ohe
