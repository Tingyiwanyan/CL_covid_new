import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(‘seaborn-whitegrid’)
# width of the bars
barWidth = 0.15
plt.figure(figsize=(10,3))
# Choose the height of the blue bars
bars1 = [0.732, 0.845, 0.863, 0.876, 0.887]
# Choose the height of the cyan bars
bars2 = [0.803, 0.916, 0.926, 0.934, 0.935]
bars3 = [0.886, 0.931, 0.941, 0.947, 0.949]
bars4 = [0.897, 0.937, 0.944, 0.952, 0.956]
bars5 = [0.906, 0.938, 0.949, 0.953, 0.954]

train_lr = [0.651,0.701,0.723,0.731,0.735]

train_rf = [0.654,0.675,0.704,0.747,0.772]

train_svm = [0.596,0.609,0.663,0.696,0.721]

train_xgb = [0.670,0.689,0.679,0.721,0.740]

CE=[0.631, 0.642,0.666,0.678,0.710]

FL=[0.656,0.671,0.683,0.723,0.756]

FL_random = [0.802,0.813,0.815,0.817,0.819]

FL_feature = [0.813,0.817,0.818,0.819,0.823]

FL_ATT = [0.815,0.820,0.825,0.827,0.833]

# The x position of bars
r1 = np.array([0,2,4,6,8])
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
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
plt.bar(r6, FL, width=barWidth, alpha=0.5, color = 'C5', edgecolor='black', capsize=7, label='FL')
plt.bar(r7, FL_random, width=barWidth, alpha=0.5, color = 'C6', edgecolor='black', capsize=7, label='FL_RANDOM')
plt.bar(r8, FL_feature, width=barWidth, alpha=0.5, color = 'C7', edgecolor='black', capsize=7, label='FL_FEATURE')
plt.bar(r9, FL_ATT, width=barWidth, alpha=0.5, color = 'C8', edgecolor='black', capsize=7, label='FL_ATTRIBUTE')
# general layout
plt.xticks([r + 4*barWidth for r in r1], ['1%', '5%', '10%','15%','20%'])
plt.ylabel('AUC')
plt.xlabel('Training set positive label percentage')
plt.ylim(0.5, 1)
#plt.legend(loc = “lower right”)
plt.legend(loc=2, bbox_to_anchor=(0.0,1.0),borderaxespad = 0.5)