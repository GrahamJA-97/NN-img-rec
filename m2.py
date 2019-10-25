#    name: Jake Graham and Chris Schulz
#    model number: 2
# -------------------------------------------------


import numpy as np
from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras.models import load_model
from sklearn import preprocessing

from preM2 import processTestData
import argparse
import time


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
    np.random.seed(1671)

    parms = parseArguments()

    X_test = np.load("MNIST_PA2/MNIST_X_test_1.npy")
    y_test = np.load("MNIST_PA2/MNIST_y_test_1.npy")

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)
    NB_CLASSES = 10

    # Pre-processing of our training data is done here to be consistent. Our pre-processing function in preM2.py is
    # defined for pre-processing the test data.
    # here we reshape and normalize our training data.
    (X_test, y_test) = processTestData(X_test, y_test)
    X_train = X_train.reshape(60000, 28, 28, 1)
    y_ohe = np_utils.to_categorical(y_train, NB_CLASSES)

    # defining constants for building our model.
    FEATURES = X_train.shape[1]
    VERBOSE = 0
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 100
    NB_EPOCHS = 10

    print('KERA modeling build starting...')
    # Build your model here
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NB_CLASSES, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='softmax'))

    feat_array = np.array(model.count_params())
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    
    #collect time as model is built one epoch at a time 
    time_array = np.empty(NB_EPOCHS)
    acc_array = np.empty(NB_EPOCHS)
    start_time = time.time()
    for i in range(NB_EPOCHS):
        model.fit(X_train, y_ohe, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
        time_array[i] = time.time() - start_time
        score = model.evaluate(X_test, y_test, verbose=VERBOSE)
        acc_array[i] = score[1]
    np.save('data/m2_time.npy', time_array)
    np.save('data/m2_acc.npy', acc_array)
    np.save('data/m2_params.npy', feat_array)
    
    print('Test loss:', score[0], 'Test accuracy:', score[1])

    # save your model
    model.save('m2.h5')


if __name__ == '__main__':
    main()
