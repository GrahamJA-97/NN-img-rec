#    name: Jake Graham and Chris Schulz
#    model number: 1
# -------------------------------------------------


import numpy as np
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn import preprocessing


from preM1 import processTestData
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()


def main():

    print("You have TensorFlow version", tf.__version__)
    np.random.seed(1671)

    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    # Pre-processing of our training data is done here to be consistent. Our pre-processing function in pa2pre.py is
    # defined for pre-processing the test data.
    # here we reshape and normalize our training data.

    X_train = X_train.reshape(60000, 784)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    NB_CLASSES = 10

    # we then make our training labels into a one hot encoding.
    y_ohe = np_utils.to_categorical(y_train, NB_CLASSES)

    X_test = np.load("MNIST_PA2/MNIST_X_test_1.npy")
    y_test = np.load("MNIST_PA2/MNIST_y_test_1.npy")

    (X_test, y_test) = processTestData(X_test, y_test)

    # defining constants for building our model.
    FEATURES = X_train.shape[1]
    VERBOSE = 0
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 100
    NB_EPOCHS = 100

    print('KERA modeling build starting...')
    ## Build your model here
    model = Sequential()
    model.add(Dense(NB_CLASSES, input_shape=(FEATURES, )))
    model.add(Activation('softmax'))

    feat_array = np.array(model.count_params())
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    # Collect data as model is built one epoch at a time 
    data_array = np.zeros([NB_EPOCHS, 4])
    start_time = time.time()
    for i in range(NB_EPOCHS):
        hist = model.fit(X_train, y_ohe, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
        data_array[i][0] = time.time() - start_time
        data_array[i][1] = hist.history['acc'][0] #training
        data_array[i][2] = hist.history['val_acc'][0] #validation
        score = model.evaluate(X_test, y_test, verbose=VERBOSE)
        data_array[i][3] = score[1] #test
        print("Epoch " + str(i + 1) + " of " + str(NB_EPOCHS) + " Completed")
    np.save('data/m1_params.npy', feat_array)
    np.save('data/m1_data.npy', data_array)

    print('Test loss:', score[0], 'Test accuracy:', score[1])

    ## save your model
    model.save(parms.outModelFile)

if __name__ == '__main__':
    main()
