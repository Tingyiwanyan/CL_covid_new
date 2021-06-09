import numpy as np
import matplotlib.pyplot as plt


death_hour = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153]

mortality_risk_fl = [0.13403179, 0.13403179, 0.16625437, 0.15813823, 0.18313171,
       0.25009975, 0.25275066, 0.21935149, 0.24090539, 0.30861455,
       0.5024046 , 0.54917365, 0.3594196 , 0.22468865, 0.3630243 ,
       0.37385952, 0.41227987, 0.34867752, 0.31556228, 0.1738695 ,
       0.3075566 , 0.28814125, 0.34813476, 0.48277915, 0.52159953,
       0.24201185, 0.16124271, 0.1966772 , 0.23361224, 0.36757416,
       0.36112446, 0.23960765, 0.18265711, 0.20585357, 0.194946  ,
       0.26015183, 0.24914022, 0.38658398, 0.2545314 , 0.32446554,
       0.3418433 , 0.35185704, 0.41917792, 0.35703218, 0.37663406,
       0.36488816, 0.15411682, 0.22449827, 0.13652968, 0.36046815,
       0.25918403, 0.33242002, 0.32842824, 0.26949343, 0.3243549 ,
       0.2799929 , 0.3163072 , 0.30187133, 0.38311377, 0.4768512 ,
       0.4788882 , 0.549358  , 0.59392095, 0.5955251 , 0.45493588,
       0.28440714, 0.21238466, 0.13796735, 0.38991606, 0.5122719 ,
       0.20978722, 0.44940576, 0.65575373, 0.629242  , 0.44505876,
       0.39498338, 0.29559523, 0.3142101 , 0.28790364, 0.25482008,
       0.32038492, 0.2807293 , 0.30043033, 0.43057775, 0.2925979 ,
       0.35141212, 0.32176664, 0.3358934 , 0.38903233, 0.22372498,
       0.19386826, 0.202293  , 0.21434425, 0.22430432, 0.24201445,
       0.14701164, 0.17706163, 0.3708965 , 0.4122457 , 0.32512403,
       0.33604947, 0.6132779 , 0.92379963, 0.78675836, 0.8000429 ,
       0.8010761 , 0.7597471 , 0.6398401 , 0.42021888, 0.34988263,
       0.4632643 , 0.48448312, 0.66922516, 0.75378406, 0.6112865 ,
       0.44571117, 0.33445656, 0.17133248, 0.10214157, 0.01711723,
       0.19311273, 0.19794345, 0.19139273, 0.2835336 , 0.26020297,
       0.33207545, 0.42431653, 0.49597102, 0.17654766, 0.23777005,
       0.3031748 , 0.24211258, 0.26994187, 0.3392582 , 0.34708455,
       0.37135297, 0.55755985, 0.4775655 , 0.18546076, 0.30976132,
       0.18845251, 0.29933774, 0.5019388 , 0.68320584, 0.43851817,
       0.37543032, 0.5409354 , 0.8675343 , 0.89041084, 0.9847478 ,
       0.9742613 , 0.98471457, 0.9726256 , 0.9799303]

mortality_risk_fl_random = [0.14508702, 0.14508702, 0.16732986, 0.15880188, 0.18332492,
       0.22207472, 0.23859905, 0.22896247, 0.2447164 , 0.29183617,
       0.481241  , 0.49161386, 0.33259058, 0.23449439, 0.33192712,
       0.36300465, 0.45039278, 0.36009687, 0.25937727, 0.12408702,
       0.32104015, 0.3276957 , 0.35650504, 0.45730892, 0.47188988,
       0.2289524 , 0.18015788, 0.21706095, 0.22659901, 0.37600574,
       0.34324816, 0.24130233, 0.18946311, 0.20962243, 0.19029507,
       0.22522119, 0.20378388, 0.32977712, 0.2696839 , 0.3300475 ,
       0.3486611 , 0.35102627, 0.38146308, 0.3645236 , 0.39760536,
       0.40062308, 0.16904703, 0.19338982, 0.11935062, 0.38423184,
       0.28306258, 0.33100984, 0.33913594, 0.32437629, 0.31020772,
       0.3029807 , 0.3208023 , 0.308206  , 0.40609655, 0.46964523,
       0.4772728 , 0.5482088 , 0.6217883 , 0.6085149 , 0.3901099 ,
       0.31601527, 0.2000752 , 0.06657356, 0.420544  , 0.5190261 ,
       0.21980037, 0.472419  , 0.7132173 , 0.6659324 , 0.44661972,
       0.36379594, 0.29010156, 0.32707962, 0.29179838, 0.25551   ,
       0.28462592, 0.30885106, 0.30006516, 0.3804609 , 0.24933165,
       0.3595842 , 0.3468286 , 0.35195106, 0.405694  , 0.23281555,
       0.21826145, 0.20896189, 0.21006249, 0.21567495, 0.23913884,
       0.14220797, 0.17805994, 0.38283056, 0.37938213, 0.30366412,
       0.31296787, 0.5566226 , 0.8545746 , 0.7927009 , 0.7919828 ,
       0.79726297, 0.77685523, 0.581064  , 0.42128143, 0.35056686,
       0.3720338 , 0.4224476 , 0.66339916, 0.6853524 , 0.5420539 ,
       0.43052018, 0.3877999 , 0.21566062, 0.12012776, 0.03034582,
       0.18069422, 0.21634278, 0.21225397, 0.42003432, 0.3059365 ,
       0.24596728, 0.40153635, 0.36629358, 0.18396473, 0.32669127,
       0.28912207, 0.21040808, 0.29141453, 0.31491593, 0.3209716 ,
       0.3667808 , 0.514488  , 0.383268  , 0.15847225, 0.27551487,
       0.24216653, 0.38375944, 0.51113546, 0.68612283, 0.4972584 ,
       0.3716217 , 0.57278055, 0.8524979 , 0.8430824 , 0.945047  ,
       0.92629325, 0.93682444, 0.9559975 , 0.97239584]

mortality_risk_fl_attribute = [0.14508702, 0.14508702, 0.16732986, 0.15880188, 0.18332492,
       0.22207472, 0.23859905, 0.22896247, 0.2447164 , 0.29183617,
       0.481241  , 0.49161386, 0.33259058, 0.23449439, 0.33192712,
       0.36300465, 0.45039278, 0.36009687, 0.25937727, 0.12408702,
       0.32104015, 0.3276957 , 0.35650504, 0.45730892, 0.47188988,
       0.2289524 , 0.18015788, 0.21706095, 0.22659901, 0.37600574,
       0.34324816, 0.24130233, 0.18946311, 0.20962243, 0.19029507,
       0.22522119, 0.20378388, 0.32977712, 0.2696839 , 0.3300475 ,
       0.3486611 , 0.35102627, 0.38146308, 0.3645236 , 0.39760536,
       0.40062308, 0.16904703, 0.19338982, 0.11935062, 0.38423184,
       0.28306258, 0.33100984, 0.33913594, 0.32437629, 0.31020772,
       0.3029807 , 0.3208023 , 0.308206  , 0.40609655, 0.46964523,
       0.4772728 , 0.5482088 , 0.6217883 , 0.6085149 , 0.3901099 ,
       0.31601527, 0.2000752 , 0.06657356, 0.420544  , 0.5190261 ,
       0.21980037, 0.472419  , 0.7132173 , 0.6659324 , 0.44661972,
       0.36379594, 0.29010156, 0.32707962, 0.29179838, 0.25551   ,
       0.28462592, 0.30885106, 0.30006516, 0.3804609 , 0.24933165,
       0.3595842 , 0.3468286 , 0.35195106, 0.405694  , 0.23281555,
       0.21826145, 0.20896189, 0.21006249, 0.21567495, 0.23913884,
       0.14220797, 0.17805994, 0.38283056, 0.37938213, 0.30366412,
       0.31296787, 0.5566226 , 0.8545746 , 0.7927009 , 0.7919828 ,
       0.79726297, 0.77685523, 0.581064  , 0.42128143, 0.35056686,
       0.3720338 , 0.4224476 , 0.66339916, 0.6853524 , 0.5420539 ,
       0.43052018, 0.3877999 , 0.21566062, 0.12012776, 0.03034582,
       0.18069422, 0.21634278, 0.21225397, 0.42003432, 0.3059365 ,
       0.24596728, 0.40153635, 0.36629358, 0.18396473, 0.32669127,
       0.28912207, 0.21040808, 0.29141453, 0.31491593, 0.3209716 ,
       0.3667808 , 0.514488  , 0.383268  , 0.15847225, 0.27551487,
       0.24216653, 0.38375944, 0.51113546, 0.68612283, 0.4972584 ,
       0.3716217 , 0.57278055, 0.8524979 , 0.8430824 , 0.945047  ,
       0.92629325, 0.93682444, 0.9559975 , 0.97239584]


plt.xlabel("hour")
plt.ylabel("mortality risk")
plt.title("Real Time Mortality Risk", fontsize=14)
#plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
#x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(np.array(death_hour)*4, mortality_risk_fl, color='green', linewidth=1, linestyle='dashed',label='FL')


plt.plot(np.array(death_hour)*4,mortality_risk_fl_random,color='blue',linestyle='dashed',label='FL+Random')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(np.array(death_hour)*4,mortality_risk_fl_attribute,color='violet',linestyle='dashed',label='FL+attribute')


plt.legend(loc='lower right')
plt.show()