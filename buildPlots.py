#    name: Jake Graham and Chris Schulz
#    building the required plots
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
# Model 1 data
m1_params = np.load('data/m1_params.npy')
m1_time = np.load('data/m1_time.npy')
m1_acc = np.load('data/m1_acc.npy')
# Model 2 data
m2_params = np.load('data/m2_params.npy')
m2_time = np.load('data/m2_time.npy')
m2_acc = np.load('data/m2_acc.npy')
# Model 3 data
m3_params = np.load('data/m3_params.npy')
m3_time = np.load('data/m3_time.npy')
m3_acc = np.load('data/m3_acc.npy')



########################################## Plot 1
MAX_EPOCHS = max([m1_time.shape[0], m2_time.shape[0], m3_time.shape[0]])
# x = np.arange(0, MAX_EPOCHS, 1)
x = np.arange(0, m1_time.shape[0], 1)
plt.plot(x, m1_time)
x = np.arange(0, m2_time.shape[0], 1)
plt.plot(x, m2_time)
x = np.arange(0, m3_time.shape[0], 1)
plt.plot(x, m3_time)

plt.xlabel()
plt.ylabel()
plt.grid(True)
plt.show()
