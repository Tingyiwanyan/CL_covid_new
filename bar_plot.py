import numpy as np
import matplotlib.pyplot as plt

size = [32,64,128,256]

FL_random = [0.802,0.806,0.815,0.816]

FL_feature = [0.805,0.809,0.823,0.824]

FL_ATT = [0.817,0.827,0.834,0.835]

plt.xlabel("Batch_size")
plt.ylabel("AUC")
plt.title("Prediction Performance", fontsize=14)
#plt.xlim(0,21)
plt.ylim(0.7, 0.9)
#x = [0.0, 1.0]

#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
#plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(size,FL_random,"x",color='lime',linestyle='dashed',linewidth=1,label='FL_RANDOM')
plt.plot(size,FL_feature,"x",color='blue',linestyle='dashed',linewidth=1,label='FL_FEATURE')
plt.plot(size,FL_ATT,"x",color='purple',linestyle='dashed',linewidth=1,label='FL_ATT')

plt.legend(loc='lower left')
plt.show()