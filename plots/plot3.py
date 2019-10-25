#    name: Jake Graham and Chris Schulz
#    Plot # 3
#       Scatter plot of parameters vs accuracy
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
# Load Model data
m1_data = np.load('../data/m1_data.npy')
m2_data = np.load('../data/m2_data.npy')
m3_data = np.load('../data/m3_data.npy')
m1_train = m1_data[-1, 1]
m1_val = m1_data[-1, 2]
m2_train = m2_data[-1, 1]
m2_val = m2_data[-1, 2]
m3_train = m3_data[-1, 1]
m3_val = m3_data[-1, 2]
m1_params = np.load('../data/m1_params.npy')
m2_params = np.load('../data/m2_params.npy')
m3_params = np.load('../data/m3_params.npy')

# # Plot the three lines
# plt.plot(np.arange(0, m1.shape[0], 1), m1, label='Model 1')
# plt.plot(np.arange(0, m2.shape[0], 1), m2, label='Model 2')
# plt.plot(np.arange(0, m3.shape[0], 1), m3, label='Model 3')

# # Add a title and labels for each of the axis 
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy of the Model(%)')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()

