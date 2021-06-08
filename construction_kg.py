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
        self.crucial_vital = ['CAC - BLOOD PRESSURE', 'CAC - TEMPERATURE', 'CAC - PULSE OXIMETRY',
                              'CAC - RESPIRATIONS', 'CAC - PULSE', 'CAC - HEIGHT', 'CAC - WEIGHT/SCALE']
        self.cricial_lab = ['ALBUMIN', 'ALKPHOS', 'ALT', 'AMYLASE', 'AGAP', 'PTT', 'AST', \
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
        for i in self.cricial_lab:
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

        for i in self.mrn_covid:
            index_reg = np.where(i==self.reg_ar[:,0])[0][0]
            index_vital = np.where(i==self.vital_sign_ar[:,0])[0]
            index_lab = np.where(i==self.lab_ar[:,0])[0]
            self.check =index_reg
            if self.reg_ar[index_reg,16] == '20' or self.reg_ar[index_reg,16] == 'EXP':
                print("found 20")
                death_flag = 1
                death_time = self.reg_ar[index_reg,15]
            elif not np.isnan(self.reg_ar[index_reg,14]):
                death_flag = 1
                death_time = self.reg_ar[index_reg,14]
            else:
                death_flag = 0
                death_time = 0
            if death_flag == 1:
                if np.isnan(death_time):
                    continue
            if not death_time == 0:
                death_time = datetime.datetime.fromtimestamp(death_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                self.in_time = death_time.split(' ')
                in_date = [np.int(i) for i in self.in_time[0].split('-')]
                in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
                self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
                in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
                total_in_time_value = in_date_value + in_time_value
                self.death_value = total_in_time_value
            else:
                self.death_value = 0

            admit_date = self.reg_ar[index_reg,3]
            self.check_date = admit_date
            admit_date = datetime.datetime.fromtimestamp(admit_date/1000).strftime('%Y-%m-%d %H:%M:%S')
            if i not in self.dic_patient.keys():
                self.dic_patient[i] = {}
                self.dic_patient[i]['death_flag'] = death_flag
                self.dic_patient[i]['death_time'] = death_time
                self.dic_patient[i]['admit_date'] = admit_date

            if death_flag == 1:
                self.death_count += 1
            """
            time value for admit
            """
            self.in_time = admit_date.split(' ')
            in_date = [np.int(i) for i in self.in_time[0].split('-')]
            in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
            self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
            in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
            total_in_time_value = in_date_value + in_time_value
            self.admit_value = total_in_time_value

            if not self.death_value == 0:
                self.death_hour = np.int(np.floor(np.float((self.death_value-self.admit_value)/60)))
                self.dic_patient[i]['death_hour'] = self.death_hour

            for k in index_lab:
                obv_id = self.labtest_ar[k][2]
                value = self.labtest_ar[k][3]
                self.check_data_lab = self.labtest_ar[k][4]
                date_year_value_lab = float(str(self.labtest_ar[k][4])[0:4])*365
                date_day_value_lab = float(str(self.check_data_lab)[4:6])*30+float(str(self.check_data_lab)[6:8])
                date_value_lab = (date_year_value_lab+date_day_value_lab)*24*60
                date_time_value_lab = float(str(self.check_data_lab)[8:10])*60+float(str(self.check_data_lab)[10:12])
                self.total_time_value_lab = date_value_lab+date_time_value_lab
                self.dic_patient[i].setdefault('lab_time_check',[]).append(self.check_data_lab)
                if obv_id in self.cricial_lab:
                    #category = self.dic_lab_category[obv_id]
                    self.prior_time = np.int(np.floor(np.float((self.total_time_value_lab-self.admit_value)/60)))
                    if self.prior_time < 0:
                        continue
                    if self.prior_time > self.death_hour:
                        continue
                    try:
                        value = float(value)
                    except:
                        continue
                    if not value == value:
                        continue

                    self.dic_lab[obv_id].setdefault('lab_value_patient',[]).append(value)
                    if self.prior_time not in self.dic_patient[i]['prior_time_lab']:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time]={}
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category,[]).append(value)
                    else:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category,[]).append(value)

            for j in index_vital:
                obv_id = self.vital_sign_ar[j][2]
                if obv_id in self.crucial_vital:
                    self.check_data = self.vital_sign_ar[j][4]
                    self.dic_patient[i].setdefault('time_capture', []).append(self.check_data)
                    date_year_value = float(str(self.vital_sign_ar[j][4])[0:4]) * 365
                    date_day_value = float(str(self.check_data)[4:6]) * 30 + float(str(self.check_data)[6:8])
                    date_value = (date_year_value + date_day_value) * 24 * 60
                    date_time_value = float(str(self.check_data)[8:10]) * 60 + float(str(self.check_data)[10:12])
                    total_time_value = date_value + date_time_value
                    self.prior_time = np.int(np.floor(np.float((total_time_value - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    if self.prior_time > self.death_hour:
                        continue
                    if obv_id == 'CAC - BLOOD PRESSURE':
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        self.check_value_presure = self.vital_sign_ar[j][3]
                        try:
                            value = self.vital_sign_ar[j][3].split('/')
                        except:
                            continue
                        if self.check_value_presure == '""':
                            continue
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
                        if self.check_value == '""':
                            continue
                        value = float(self.vital_sign_ar[j][3])
                        if np.isnan(value):
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                                value)
                        self.dic_vital[obv_id].setdefault('value', []).append(value)






if __name__ == "__main__":
    kg = kg_construction()
    kg.read_csv()
    kg.create_kg_dic()
