import numpy as np
import matplotlib.pyplot as plt

# Make a random dataset:
height = [0.811, 0.821, 0.827, 0.825, 0.823]
bars = ('1', '3', '5', '9', '15')
y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, height,width=0.6,color='blue')

# Create names on the x-axis
plt.xticks(y_pos, bars)

plt.xlabel("k")
plt.ylabel("AUC")
#plt.title("Prediction Performance on Different k", fontsize=14)
#plt.xlim(0,21)
plt.ylim(0.8, 0.84)

# Show graphic
plt.show()
plt.show()