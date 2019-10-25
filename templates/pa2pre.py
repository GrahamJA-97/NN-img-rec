#    name: pa2pre.py
# purpose: Student's add code to preprocessing of the data

# Recall that any preprocessing you do on your training
# data, you must also do on any future data you want to
# predict.  This file allows you to perform any
# preprocessing you need on my undisclosed test data

NB_CLASSES=10
import numpy as np
from keras.utils import np_utils

def processTestData(X, y):

    # X preprocessing goes here -- students optionally complete

    # y preprocessing goes here.  y_test becomes a ohe
    y_ohe = np_utils.to_categorical (y, NB_CLASSES)
    return X, y_ohe
