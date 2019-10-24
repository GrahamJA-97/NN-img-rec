#    name: Jake Graham and Chris Schulz
#    model number: 3
# -------------------------------------------------


import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils import np_utils
from keras.models import load_model
from sklearn import preprocessing

from preM3 import processTestData
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

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    X_test = np.load("MNIST_PA2/MNIST_X_test_1.npy")
    y_test = np.load("MNIST_PA2/MNIST_y_test_1.npy")

    (X_test, y_test) = processTestData(X_test, y_test)

    # Pre-processing of our training data is done here to be consistent. Our pre-processing function in pa2pre.py is
    # defined for pre-processing the test data.
    # here we reshape and normalize our training data.

    X_train = X_train.reshape(60000, 784)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    NB_CLASSES = 10

    # we then make our training labels into a one hot encoding.
    y_ohe = np_utils.to_categorical(y_train, NB_CLASSES)

    FEATURES = X_train.shape[1]
    feat_array = np.array(FEATURES)
    VERBOSE = 0
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 100
    NB_EPOCHS = 50

    print('KERA modeling build starting...')
    # Build your model here

    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(FEATURES, )))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
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
    np.save('m3_time.npy', time_array)
    np.save('m3_acc.npy', acc_array)
    np.save('m3_feat.npy', feat_array)
    
    print('Test loss:', score[0], 'Test accuracy:', score[1])

    #got .984 accurate with 300 epochs and 7 layers.
    # got .984899 with 50 epocs first 3 layers.

    # save your model
    model.save(str(parms.outModelFile))

if __name__ == '__main__':
    main()
