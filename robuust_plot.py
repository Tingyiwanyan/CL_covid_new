import numpy as np
import matplotlib.pyplot as plt

size = [345,945,1945,2945,3405]

train_lr = [0.695,0.743,0.741,0.753,0.754]

train_rf = [0.711,0.721,0.772,0.777,0.778]

train_svm = [0.690,0.719,0.751,0.751,0.760]

train_xgb = [0.684,0.689,0.729,0.749,0.758]

CE=[0.710]

FL=[0.777]

FL_random = [0.761,0.781,0.803,0.806]

FL_feature = [0.766,0.786,0.812,0.815]

FL_ATT = [0.774,0.803,0.815,0.827]

plt.xlabel("Traing Size")
plt.ylabel("AUC")
plt.title("Prediction Performance on Various Sample Sizes in the same Test Set", fontsize=14)
#plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]

plt.plot(recall_total_lr, precision_total_lr, color='green', linewidth=2, linestyle='dashed',label='LR(AUPRC=0.524)')
plt.plot(recall_total_rf, precision_total_rf,color='blue',linestyle='dashed',linewidth=2,label='RF(AUPRC=0.593)')
plt.plot(recall_total_svm,precision_total_svm,color='violet',linestyle='dashed',linewidth=2,label='SVM(AUPRC=0.560)')
plt.plot(recall_total_xgb,precision_total_xgb,color='red',linestyle='dashed',linewidth=2,label='XGB(AUPRC=0.563)')
#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(recall_total_fl_random,precision_total_fl_random,color='gray',linestyle='solid',linewidth=2,label='FL_RANDOM(AUPRC=0.620)')
plt.plot(recall_total_fl_feature,precision_total_fl_feature,color='pink',linestyle='solid',linewidth=2,label='FL_FEATURE(AUPRC=0.627)')
plt.plot(recall_total_fl_att,precision_total_fl_att,color='purple',linestyle='solid',linewidth=2,label='FL_ATT(AUPRC=0.655)')

plt.legend(loc='lower left')
plt.show()