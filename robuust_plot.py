import numpy as np
import matplotlib.pyplot as plt

size = [345,945,1945,2945,3405]

train_lr = [0.695,0.743,0.741,0.753,0.754]

train_rf = [0.711,0.721,0.772,0.777,0.778]

train_svm = [0.690,0.719,0.751,0.751,0.760]

train_xgb = [0.684,0.689,0.729,0.749,0.758]

CE=[0.668, 0.672,0.694,0.701,0.710]

FL=[0.777]

FL_random = [0.761,0.781,0.803,0.806]

FL_feature = [0.766,0.786,0.812,0.815]

FL_ATT = [0.774,0.803,0.815,0.827]

plt.xlabel("Traing Size")
plt.ylabel("AUC")
plt.title("Prediction Performance", fontsize=14)
plt.xlim(300, 3500)
plt.ylim(0.5, 0.9)
#x = [0.0, 1.0]

plt.plot(size, train_lr, "s-",color='green', linewidth=2, linestyle='dashed',label='LR')

plt.legend(loc='lower left')
plt.show()