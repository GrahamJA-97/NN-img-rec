# Print a bar chart with groups

import numpy as np
import matplotlib.pyplot as plt

# set height of bar
# length of these lists determine the number
# of groups (they must all be the same length)
bars1 = [.92] # bars for model 1
bars2 = [.98] # bars for model 2
bars3 = [.98] # bars for model 3
bars4 = [.91] # bars for noise model 1
bars5 = [.98] # bars for noise model 2
bars6 = [.98] # bars for noise model 3


# set width of bar. To work and supply some padding
# the number of groups times barWidth must be
# a little less than 1 (since the next group
# will start at 1, then 2, etc).

barWidth = 1
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='m1.h5')
plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='white', label='m2.h5')
plt.bar(r3, bars3, color='orange', width=barWidth, edgecolor='white', label='m3.h5')
plt.bar(r4, bars4, color='red', width=barWidth, edgecolor='white', label='m1_n20.h5')
plt.bar(r5, bars5, color='purple', width=barWidth, edgecolor='white', label='m2_n20.h5')
plt.bar(r6, bars6, color='yellow', width=barWidth, edgecolor='white', label='m3_n20.h5')

objects = ('m1.h5', 'm2.h5', 'm3.h5', 'm1_n20.h5', 'm2_n20.h5', 'm3_n20.h5')
arrange = np.arange(len(objects))

# Add xticks on the middle of the group bars
plt.xlabel('Accuracy of all 6 models with Original Test Data', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['m1.h5', 'm2.h5', 'm3.h5', 'm1_n20.h5', 'm2_n20.h5', 'm3_n20.h5'])
plt.xticks(arrange, objects)
# Create legend & Show graphic
plt.ylabel('Accuracy of the Models (%)')
plt.show()
#plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf