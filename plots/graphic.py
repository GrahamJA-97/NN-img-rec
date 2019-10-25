#    name: Jake Graham and Chris Schulz
#    Graphic
#       
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
# Load Model data
x = np.load('../Noise_files/extract/n40_data/MNIST_X_train_1_n40.npy')

plt.imshow(x[1, :])

plt.show()

