import numpy as np
import random
import math
import time
import pandas as pd
import json
import datetime

class kg_construction():
    def __init__(self):
        file_path = '/datadrive/tingyi_wanyan/Registry_2020-10-15'
        self.reg = file_path + '/registry.csv'
        self.covid_lab = file_path + '/covid19LabTest.csv'
        self.lab = file_path + '/Lab.csv'
        self.vital = file_path + '/vitals.csv'
        self.lab_comb = '/datadrive/tingyi_wanyan/lab_mapping_comb.csv'
        self.file_path_comorbidity = '/home/tingyi.wanyan/comorbidity_matrix_20200710.csv'


    def read_csv(self):
        self.registry = pd.read_csv(self.reg)
        self.covid_labtest = pd.read_csv(self.covid_lab)
        self.labtest = pd.read_csv(self.lab)
        self.vital_sign = pd.read_csv(self.vital)
        # self.comorbidity = pd.read_csv(self.file_path_comorbidity)
        self.lab_comb = pd.read_csv(self.lab_comb)
        self.reg_ar = np.array(self.registry)
        self.covid_ar = np.array(self.covid_labtest)
        self.labtest_ar = np.array(self.labtest)
        self.vital_sign_ar = np.array(self.vital_sign)
        self.lab_comb_ar = np.array(self.lab_comb)


    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_vital = {}
        self.dic_lab = {}
        self.dic_lab_category = {}
        self.dic_demographic = {}
        self.dic_race = {}
        self.death_neighbor = []
        self.live_neighbor = []
        self.crucial_vital = ['CAC - BLOOD PRESSURE', 'CAC - TEMPERATURE', 'CAC - PULSE OXIMETRY',
                              'CAC - RESPIRATIONS', 'CAC - PULSE', 'CAC - HEIGHT', 'CAC - WEIGHT/SCALE']
        self.crucial_lab = ['ALBUMIN', 'ALKPHOS', 'ALT', 'AMYLASE', 'AGAP', 'PTT', 'AST', \
                           'ATYPLYMPH', 'BANDS', 'BASOPHIL_PERC', 'BASOPHIL', 'DBILIRUBIN', \
                           'TBILIRUBIN', 'BLASTS', 'BNP', 'BUN', 'CRP', 'CALCIUMIONIZED_A', \
                           'CALCIUMIONIZED', 'CALCIUM', 'CHLORIDE', 'CPK', 'CKMB', 'BICARB', \
                           'CREATININE', 'DDIMER', 'EOSINO_PERC', 'EOSINO', 'FERRITIN', 'FIBRINOGEN', \
                           'GLUCOSE', 'GLUCOSE_A', 'BICARB_A', 'HCT', 'HCT_A', 'HGB', 'HGB_A', 'INR', \
                           'IL6', 'IRON', 'KETONE', 'LACTATE_A', 'LACTATE', 'LDH', 'LYMPHO_PERC', \
                           'LYMPHO', 'MCHC', 'MCV', 'MPV', 'MONO_PERC', 'MONO', 'NEUTRO_PERC', \
                           'NEUTRO', 'O2SAT', 'O2SAT_A', 'PCO2_A', 'PCO2', 'PH_A', 'PH', 'PLT', \
                           'PO2_A', 'PO2', 'POTASSIUM', 'POTASSIUM_A', 'PT', 'PROTEIN', 'RBCCNT', \
                           'RDW', 'SODIUM', 'SODIUM_A', 'TIBC', 'TRANSFERRINSAT', 'TROPONINI', 'URICACID', \
                           'CHLORIDE_A', 'WBC']
        self.index_covid = np.where(self.covid_ar[:,7]=='DETECTED')[0]
        self.mrn_covid = np.unique(self.covid_ar[self.index_covid,:][:,0])
        self.death_count = 0
        index_lab = 0
        for i in self.crucial_lab:
            self.dic_lab[i] = {}
            self.dic_lab[i]['index'] = index_lab
            index_lab += 1

        index_vital = 0
        for i in self.crucial_vital:
            if i == 'CAC - BLOOD PRESSURE':
                self.dic_vital['high'] = {}
                self.dic_vital['high']['index'] = index_vital
                index_vital += 1
                self.dic_vital['low'] = {}
                self.dic_vital['low']['index'] = index_vital
                index_vital += 1
            else:
                self.dic_vital[i] = {}
                self.dic_vital[i]['index'] = index_vital
                index_vital += 1

        index_count = 0
        for i in self.mrn_covid:
            print(index_count)
            index_reg = np.where(i==self.reg_ar[:,0])[0][0]
            index_vital = np.where(i==self.vital_sign_ar[:,0])[0]
            index_lab = np.where(i==self.labtest_ar[:,0])[0]
            self.check =index_reg
            self.check_lab = index_lab
            self.check_vital = index_vital
            if self.reg_ar[index_reg,16] == '20' or self.reg_ar[index_reg,16] == 'EXP':
                #print("found 20")
                death_flag = 1
                death_time = self.reg_ar[index_reg,15]
            elif not np.isnan(self.reg_ar[index_reg,14]):
                death_flag = 1
                death_time = self.reg_ar[index_reg,14]

            else:
                death_flag = 0
                self.discharge_time = self.reg_ar[index_reg,15]
                if np.isnan(self.discharge_time):
                    continue
                death_time = self.discharge_time

            if np.isnan(death_time):
                continue
            if death_flag == 1:
                self.death_neighbor.append(i)
            else:
                self.live_neighbor.append(i)
            #if not death_time == 0:
            death_time = datetime.datetime.fromtimestamp(death_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            self.in_time_death = death_time.split(' ')
            self.in_date_death = [np.int(i) for i in self.in_time_death[0].split('-')]
            self.in_time_death = [np.int(i) for i in self.in_time_death[1].split(':')[0:-1]]
            #else:
               # self.death_value = 0

            admit_date = self.reg_ar[index_reg,3]
            self.check_date = admit_date
            admit_date = datetime.datetime.fromtimestamp(admit_date/1000).strftime('%Y-%m-%d %H:%M:%S')
            if i not in self.dic_patient.keys():
                self.dic_patient[i] = {}
                self.dic_patient[i]['prior_time_lab'] = {}
                self.dic_patient[i]['prior_time_vital'] = {}
                self.dic_patient[i]['death_flag'] = death_flag
                self.dic_patient[i]['death_time'] = death_time
                self.dic_patient[i]['admit_date'] = admit_date

            if death_flag == 1:
                self.death_count += 1
            """
            time value for admit
            """
            self.in_time_admit = admit_date.split(' ')
            self.in_date_admit = [np.int(i) for i in self.in_time_admit[0].split('-')]
            self.in_time_admit = [np.int(i) for i in self.in_time_admit[1].split(':')[0:-1]]


            difference_month = self.in_date_death[1]-self.in_date_admit[1]
            difference_day = self.in_date_death[2]-self.in_date_admit[2]
            difference_hour = self.in_time_death[0]-self.in_time_admit[0]

            death_month = difference_month * 30 * 6
            death_day = difference_day * 6
            death_hour = np.int(np.floor(difference_hour / 4))

            death_time = death_month + death_day + death_hour

            if not death_flag == 0:
                self.dic_patient[i]['death_hour'] = death_time
            else:
                self.dic_patient[i]['discharge_hour'] = death_time

            if death_time > 180:
                continue

            for k in index_lab:
                obv_id = self.labtest_ar[k][2]
                value = self.labtest_ar[k][3]
                self.check_data_lab = self.labtest_ar[k][4]
                #date_day_value_lab = float(str(self.check_data_lab)[4:6])*30+float(str(self.check_data_lab)[6:8])
                date_month_value_lab = float(str(self.check_data_lab)[4:6])
                date_day_value_lab = float(str(self.check_data_lab)[6:8])
                date_hour_value_lab = float(str(self.check_data_lab)[8:10])
                #date_value_lab = (date_year_value_lab+date_day_value_lab)*24*60
                #date_time_value_lab = float(str(self.check_data_lab)[8:10])*60+float(str(self.check_data_lab)[10:12])
                #self.total_time_value_lab = date_value_lab+date_time_value_lab
                #self.dic_patient[i].setdefault('lab_time_check',[]).append(self.check_data_lab)
                if obv_id in self.crucial_lab:
                    difference_month = date_month_value_lab - self.in_date_admit[1]
                    difference_day = date_day_value_lab - self.in_date_admit[2]
                    difference_hour = date_hour_value_lab - self.in_time_admit[0]

                    death_month = difference_month * 30 * 6
                    death_day = difference_day * 6
                    death_hour = np.int(np.floor(difference_hour / 4))

                    self.prior_time = death_month + death_day + death_hour
                    #category = self.dic_lab_category[obv_id]
                    #self.prior_time = np.int(np.floor(np.float((self.total_time_value_lab-self.admit_value)/(60*6))))
                    if self.prior_time < 0:
                        continue
                    if self.prior_time > death_time:
                        continue
                    #try:
                       # value = float(value)
                    #except:
                        #continue
                    #if not value == value:
                        #continue
                    if value == value:
                        self.dic_lab[obv_id].setdefault('lab_value_patient',[]).append(value)
                        if self.prior_time not in self.dic_patient[i]['prior_time_lab']:
                            self.dic_patient[i]['prior_time_lab'][self.prior_time]={}
                            self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(obv_id,[]).append(value)
                        else:
                            self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(obv_id,[]).append(value)
            

            for j in index_vital:
                obv_id = self.vital_sign_ar[j][2]
                if obv_id in self.crucial_vital:
                    self.check_data_vital = self.vital_sign_ar[j][4]

                    date_month_value_lab = float(str(self.check_data_vital)[4:6])
                    date_day_value_lab = float(str(self.check_data_vital)[6:8])
                    date_hour_value_lab = float(str(self.check_data_vital)[8:10])

                    difference_month = date_month_value_lab - self.in_date_admit[1]
                    difference_day = date_day_value_lab - self.in_date_admit[2]
                    difference_hour = date_hour_value_lab - self.in_time_admit[0]

                    death_month = difference_month * 30 * 6
                    death_day = difference_day * 6
                    death_hour = np.int(np.floor(difference_hour / 4))

                    self.prior_time = death_month + death_day + death_hour

                    if self.prior_time < 0:
                        continue
                    if self.prior_time > death_time:
                        continue
                    if obv_id == 'CAC - BLOOD PRESSURE':
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        self.check_value_presure = self.vital_sign_ar[j][3]
                        if not self.check_value_presure == '""':
                            try:
                                value = self.vital_sign_ar[j][3].split('/')
                            except:
                                value = 0
                            if not value == 0:
                                if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                        value[0])
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                        value[1])
                                else:
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high', []).append(
                                        value[0])
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low', []).append(
                                        value[1])
                                self.dic_vital['high'].setdefault('value', []).append(value[0])
                                self.dic_vital['low'].setdefault('value', []).append(value[1])

                    else:
                        self.check_value = self.vital_sign_ar[j][3]
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        if not self.check_value == '""':
                            value = float(self.vital_sign_ar[j][3])
                            if not np.isnan(value):
                                if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                        value)
                                else:
                                    self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                        value)
                                self.dic_vital[obv_id].setdefault('value', []).append(value)

            index_count += 1





if __name__ == "__main__":
    #kg = kg_construction()
    #kg.read_csv()
    #kg.create_kg_dic()
    #read_d = read_data_covid()
