#    name: Jake Graham and Chris Schulz
#    Plot # 1
#       Line plot of epochs vs time to build
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
# Load Model data
m1 = np.load('../data/m1_data.npy')[:, 0]
m2 = np.load('../data/m2_data.npy')[:, 0]
m3 = np.load('../data/m3_data.npy')[:, 0]

# Plot the three lines
plt.plot(np.arange(0, m1.shape[0], 1), m1, label='Model 1')
plt.plot(np.arange(0, m2.shape[0], 1), m2, label='Model 2')
plt.plot(np.arange(0, m3.shape[0], 1), m3, label='Model 3')

# Add a title and labels for each of the axis 
plt.xlabel('Epochs')
plt.ylabel('Time to build the Model (seconds)')
plt.title('Comparing # of Epochs to the Time to Build')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

