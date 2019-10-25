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

# Plot the three lines
plt.scatter(m1_params, m1_train, color='b', label='Training 1')
plt.scatter(m1_params, m1_val, color='c', label='Validation 1')
plt.scatter(m2_params, m2_train, color='r', label='Training 2')
plt.scatter(m2_params, m2_val, color='m', label='Validation 2')
plt.scatter(m3_params, m3_train, color='g', label='Training 3')
plt.scatter(m3_params, m3_val, color='y', label='Validation 3')



# Add a title and labels for each of the axis 
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy of the Model(%)')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()

