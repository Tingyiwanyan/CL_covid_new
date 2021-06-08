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

        self.index_covid = np.where(self.covid_ar[:,7]=='DETECTED')[0]
        self.mrn_covid = np.unique(self.covid_ar[self.index_covid,:][:,0])
        self.death_count = 0
        for i in self.mrn_covid:
            index_reg = np.where(i==self.reg_ar[:,0])[0][0]
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
            death_time = datetime.datetime.fromtimestamp(death_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
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




if __name__ == "__main__":
    kg = kg_construction()
    kg.read_csv()
    kg.create_kg_dic()
