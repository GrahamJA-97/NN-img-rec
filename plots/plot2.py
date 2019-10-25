#    name: Jake Graham and Chris Schulz
#    Plot # 2
#       3 Line plots of epochs vs accuracy
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
# Load Model data
m1_data = np.load('../data/m1_data.npy')
m2_data = np.load('../data/m2_data.npy')
m3_data = np.load('../data/m3_data.npy')
m1_train = m1_data[:, 1]
m1_val = m1_data[:, 2]
m2_train = m2_data[:, 1]
m2_val = m2_data[:, 2]
m3_train = m3_data[:, 1]
m3_val = m3_data[:, 2]

# make the 3 plots
plt.subplot(3, 1, 1)
plt.plot(np.arange(0, m1_train.shape[0], 1), m1_train, label='Train 1')
plt.plot(np.arange(0, m1_val.shape[0], 1), m1_val, label='Validation 1')
plt.title('Model Train vs Validation Accuracy Over Epochs')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, m2_train.shape[0], 1), m2_train, label='Train 2')
plt.plot(np.arange(0, m2_val.shape[0], 1), m2_val, label='Validation 2')
plt.ylabel('Accuracy of the Model(%)')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, m3_train.shape[0], 1), m3_train, label='Train 3')
plt.plot(np.arange(0, m3_val.shape[0], 1), m3_val, label='Validation 3')
plt.legend(loc='lower right')
plt.grid(True)

# Add a title and labels for each of the axis
plt.xlabel('Epochs')
plt.show()

