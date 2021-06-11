import numpy as np
import matplotlib.pyplot as plt


step = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

acc_ce = [0.5460084 , 0.62968487, 0.65054622, 0.67136555, 0.69468487,
       0.71294118, 0.72571429, 0.73460084, 0.73897059, 0.73915966,
       0.73882353, 0.73673319, 0.73464286, 0.73921218, 0.74576681,
       0.75152311, 0.75778361, 0.76      , 0.7622479 , 0.76084034,
       0.75818277, 0.75921218, 0.75970588, 0.76093487, 0.76284664,
       0.76593487, 0.76779412, 0.77061975, 0.7727416 , 0.77263655,
       0.76957983, 0.76455882, 0.76183824]

acc_fl = [0.5347268907563025,
 0.6179831932773109,
 0.6629621848739495,
 0.7005042016806723,
 0.7307773109243698,
 0.7463445378151259,
 0.7530042016806722,
 0.762247899159664,
 0.7690756302521008,
 0.7736554621848738,
 0.776155462184874,
 0.7772058823529411,
 0.7775840336134454,
 0.7750630252100841,
 0.7757983193277311,
 0.7805042016806722,
 0.7808613445378151,
 0.7740546218487395,
 0.7760504201680671,
 0.7769747899159664,
 0.7801470588235294,
 0.7831302521008405,
 0.784579831932773,
 0.7851050420168068,
 0.780609243697479,
 0.7798739495798321,
 0.7864075630252101,
 0.7863655462184874,
 0.7791806722689075,
 0.7789915966386554,
 0.7787815126050419,
 0.7816176470588235,
 0.7842226890756303]

acc_fl_feature = [0.53465546, 0.6040336, 0.6510084 , 0.69228992, 0.71668067,
       0.72993697, 0.73771008, 0.75256303, 0.76292017, 0.76754202,
       0.77348739, 0.77726891, 0.77697479, 0.77239496, 0.77607143,
       0.77579832, 0.77331933, 0.77462185, 0.77659664, 0.77810924,
       0.7797479 , 0.78121849, 0.78220588, 0.78136555, 0.77983193,
       0.77947479, 0.78243697, 0.78262605, 0.78029412, 0.78084034,
       0.78352941, 0.78529412, 0.78544118]

acc_fl_att = [0.55397059, 0.6344538, 0.70369748, 0.74090336, 0.7494958 ,
       0.75170168, 0.75676471, 0.7585084 , 0.7689916 , 0.78153361,
       0.78144958, 0.77939076, 0.78771008, 0.79844538, 0.81092437,
       0.80964286, 0.80418067, 0.79707983, 0.80380252, 0.81153361,
       0.81592437, 0.81308824, 0.81252101, 0.81222689, 0.81680672,
       0.81936975, 0.81781513, 0.81186975, 0.80817227, 0.81628151,
       0.81556723, 0.81607143, 0.815]

