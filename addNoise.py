#    name: addNoise.py
#  author: molloykp (Oct 2019)
# purpose: Accept a numpy npy file and add Gaussian noise
#          Parameters:

import numpy as np
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Add noise to image(s)')

    parser.add_argument('--inputFile', action='store',
                        dest='inputFile', default="", required=True,
                        help='matrix of images in npy format')
    parser.add_argument('--sigma', action='store',type=float,
                        dest='sigma', default="", required=True,
                        help='std dev used in generating noise')

    parser.add_argument('--outputFile', action='store',
                        dest='outputFile', default="", required=True,
                        help='file to store matrix with noise')

    return parser.parse_args()

def main():
    np.random.seed(1671)

    parms = parseArguments()

    inMatrix = np.load(parms.inputFile)

    # matrix must be floating point to add values
    # from the Gaussian
    inMatrix = inMatrix.astype('float32')
    inMatrix += np.random.normal(0,parms.sigma,(inMatrix.shape))
    inMatrix = inMatrix.astype('int')

    # noise may have caused values to go outside their allowable
    # range
    inMatrix[inMatrix < 0] = 0
    inMatrix[inMatrix > 255] = 255

    np.save(parms.outputFile,inMatrix)

if __name__ == '__main__':
    main()