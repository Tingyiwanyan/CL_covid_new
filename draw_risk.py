import numpy as np
import matplotlib.pyplot as plt


death_hour = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

mortality_risk_fl = [0.11235922, 0.11235922, 0.18750528, 0.2385273 , 0.2445534 ,
       0.17229636, 0.19012132, 0.31032366, 0.32832092, 0.31663856,
       0.3233032 , 0.36129284, 0.4029985 , 0.3948899 , 0.389836  ,
       0.35027933, 0.32449105, 0.43031815, 0.45599476, 0.58588815,
       0.5474469 , 0.5616107 , 0.65647817, 0.681786  , 0.68102497,
       0.65377694, 0.68689317, 0.6670625 , 0.6367486 , 0.6621304 ,
       0.74364674, 0.8046682 ]

mortality_risk_fl_random = [0.19609131, 0.19609131, 0.26063663, 0.34964898, 0.3110687 ,
       0.29455087, 0.33321768, 0.41986814, 0.4213427 , 0.4183155 ,
       0.428137  , 0.4747546 , 0.52159315, 0.5115459 , 0.48880228,
       0.45464474, 0.45974362, 0.49561647, 0.5501274 , 0.6954123 ,
       0.65649587, 0.71616864, 0.82525784, 0.83199435, 0.8446984 ,
       0.8137904 , 0.8184372 , 0.76975   , 0.728778  , 0.7679413 ,
       0.8245465 , 0.8484929 ]

mortality_risk_fl_attribute = [0.16277437, 0.16277437, 0.22073472, 0.31575546, 0.29373   ,
       0.23756881, 0.26810506, 0.31053147, 0.30026853, 0.29353878,
       0.3338364 , 0.3559694 , 0.43792522, 0.39395744, 0.3842598 ,
       0.3858591 , 0.35140413, 0.49032003, 0.435321  , 0.6185268 ,
       0.5785474 , 0.65627253, 0.7700119 , 0.7780227 , 0.80199134,
       0.7543927 , 0.7675434 , 0.75236654, 0.7004006 , 0.73200154,
       0.78929514, 0.8434022 ]


plt.xlabel("hour")
plt.ylabel("mortality risk")
plt.title("Real Time Mortality Risk", fontsize=14)
#plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
#x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(np.array(death_hour)*4, mortality_risk_fl, color='green', linewidth=1, linestyle='dashed',label='FL')


plt.plot(np.array(death_hour)*4,mortality_risk_fl_random,color='blue',linestyle='dashed',label='FL+attribute')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(np.array(death_hour)*4,mortality_risk_fl_attribute,color='violet',linestyle='dashed',label='FL+random')


plt.legend(loc='lower right')
plt.show()