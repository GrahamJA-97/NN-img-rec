#    name: Jake Graham and Chris Schulz
#    building the required plots
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
 
m1_params = np.load('m1_feat.npy')
m1_time = np.load('m1_time.npy')
m1_acc = np.load('m1_acc.npy')
# m2_time = np.load('m2_time.npy')
# m2_acc = np.load('m2_acc.npy')
# m3_time = np.load('m3_time.npy')
# m3_acc = np.load('m3_acc.npy')
# print(m1_params)


# Plot 1
# MAX_EPOCHS = max(m1_time.shape[0])
MAX_EPOCHS = m1_time.shape[0]
x = np.arange(0, MAX_EPOCHS, 1)
plt.plot(x, m1_time)

plt.grid(True)
# plt.show()
