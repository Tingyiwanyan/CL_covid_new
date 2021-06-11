import numpy as np
import matplotlib.pyplot as plt

size = [1,5,10,15,20]

train_lr = [0.651,0.701,0.723,0.731,0.754]

train_rf = [0.654,0.675,0.704,0.747,0.778]

train_svm = [0.596,0.609,0.663,0.696,0.760]

train_xgb = [0.670,0.689,0.679,0.721,0.758]

CE=[0.668, 0.672,0.694,0.701,0.710]

FL=[0.705,0.731,0.756,0.761,0.777]

FL_random = [0.761,0.781,0.803,0.806,0.815]

FL_feature = [0.766,0.786,0.812,0.815,0.820]

FL_ATT = [0.774,0.803,0.815,0.827,0.830]

plt.xlabel("Traing Size")
plt.ylabel("AUC")
plt.title("Prediction Performance", fontsize=14)
plt.xlim(300, 3500)
plt.ylim(0.5, 0.9)
#x = [0.0, 1.0]

plt.plot(size, train_lr, "D",color='green', linewidth=2, linestyle='dashed',label='LR')
plt.plot(size, train_rf,"D",color='blue',linestyle='dashed',linewidth=2,label='RF')
plt.plot(size,train_svm,"D",color='violet',linestyle='dashed',linewidth=2,label='SVM')
plt.plot(size,train_xgb,"D",color='red',linestyle='dashed',linewidth=2,label='XGB')
plt.plot(size,CE,"D",color='indigo',linestyle='dashed',linewidth=2,label='CE')
plt.plot(size,FL,"D",color='orange',linestyle='dashed',linewidth=2,label='FL')
#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
#plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(size,FL_random,"D",color='gray',linestyle='solid',linewidth=2,label='FL_RANDOM')
plt.plot(size,FL_feature,"D",color='pink',linestyle='solid',linewidth=2,label='FL_FEATURE')
plt.plot(size,FL_ATT,"D",color='purple',linestyle='solid',linewidth=2,label='FL_ATT')

plt.legend(loc='lower left')
plt.show()