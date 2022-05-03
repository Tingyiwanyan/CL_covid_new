import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(‘seaborn-whitegrid’)
# width of the bars
barWidth = 0.15
plt.figure(figsize=(10,3))
size = [345,945,1945,2945,3405]

train_lr = [0.673,0.696,0.741,0.753,0.754]

train_rf = [0.711,0.721,0.772,0.777,0.778]

train_svm = [0.690,0.719,0.751,0.751,0.760]

train_xgb = [0.684,0.689,0.729,0.749,0.758]

CE=[0.668, 0.672,0.694,0.701,0.710]

FL=[0.705,0.731,0.756,0.761,0.777]

FL_random = [0.761,0.781,0.803,0.806,0.815]

FL_feature = [0.766,0.786,0.812,0.815,0.820]

FL_ATT = [0.774,0.803,0.815,0.827,0.830]

# The x position of bars
r1 = np.array([0,2,4,6,8])
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
#r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r5]
r8 = [x + barWidth for x in r7]
r9 = [x + barWidth for x in r8]


# Create blue bars
plt.bar(r1, train_lr, width=barWidth, alpha=0.5, color ='C0', edgecolor='black',  capsize=7, label='LR')
plt.yticks(np.arange(0, 1, 0.05))
# Create cyan bars
plt.bar(r2, train_rf, width=barWidth, alpha=0.5, color = 'C1', edgecolor='black', capsize=7, label='RF')
plt.bar(r3, train_svm, width=barWidth, alpha=0.5, color = 'C2', edgecolor='black',  capsize=7, label='SVM')
plt.bar(r4, train_xgb, width=barWidth, alpha=0.5, color = 'C3', edgecolor='black', capsize=7, label='XGB')
plt.bar(r5, CE, width=barWidth, alpha=0.5, color = 'C4', edgecolor='black', capsize=7, label='CE')
#plt.bar(r6, FL, width=barWidth, alpha=0.5, color = 'C5', edgecolor='black', capsize=7, label='FL')
plt.bar(r7, FL_random, width=barWidth, alpha=0.5, color = 'C6', edgecolor='black', capsize=7, label='FL_RANDOM')
plt.bar(r8, FL_feature, width=barWidth, alpha=0.5, color = 'C7', edgecolor='black', capsize=7, label='FL_FEATURE')
plt.bar(r9, FL_ATT, width=barWidth, alpha=0.5, color = 'C8', edgecolor='black', capsize=7, label='FL_ATTRIBUTE')
# general layout
plt.xticks([r + 4*barWidth for r in r1], ['345', '945', '1945','2945','3405'])
plt.ylabel('AUC')
plt.xlabel('Traing Data Size')
plt.ylim(0.55, 0.85)
#plt.legend(loc = “lower right”)
plt.legend(loc=2, bbox_to_anchor=(0.0,1.0),borderaxespad = 0.5)