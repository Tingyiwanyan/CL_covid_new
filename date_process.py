import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir


class read_data_covid():
    """
    Loading data, mean and std are pre-computed
    """
    def __init__(self):
        self.file_path_patient = "/datadrive/tingyi_wanyan/covid_data/dic_patient.json"
        self.file_path_lab = "/datadrive/tingyi_wanyan/covid_data/dic_lab.json"
        self.file_path_vital = "/datadrive/tingyi_wanyan/covid_data/dic_vital.json"
        file_patient = open(self.file_path_patient,"r")
        file_lab = open(self.file_path_lab,"r")
        file_vital = open(self.file_path_vital,"r")
        self.dic_patient = json.load(file_patient)
        self.dic_lab = json.load(file_lab)
        self.dic_vital = json.load(file_vital)

        self.time_sequence = 4
        self.predict_window = 3

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

        for i in self.crucial_lab:
            values = []
            if not 'lab_value_patient' in self.dic_lab[i].keys():
                self.dic_lab.pop(i)
                continue
            for k in self.dic_lab[i]['lab_value_patient']:
                try:
                    k_ = np.float(k)
                    values.append(k_)
                except:
                    continue
            up_value = np.percentile(values,95)
            low_value = np.percentile(values,5)
            values_ = [k for k in values if k < up_value and k > low_value]

            mean_lab = np.mean(values_)
            std_lab = np.std(values_)
            self.dic_lab[i]['mean_value'] = mean_lab
            self.dic_lab[i]['std'] = std_lab

        for i in self.crucial_lab:
            if i in self.dic_lab.keys():
                if np.isnan(self.dic_lab[i]['mean_value']):
                    self.dic_lab.pop(i)
        
        for i in self.dic_vital.keys():
            values = []
            for k in self.dic_vital[i]['value']:
                try:
                    k_ = np.float(k)
                    values.append(k_)
                except:
                    continue
            up_value = np.percentile(values, 95)
            low_value = np.percentile(values, 5)
            values_ = [k for k in values if k < up_value and k > low_value]
            mean = np.mean(values_)
            std = np.std(values_)
            self.dic_vital[i]['mean_value'] = mean
            self.dic_vital[i]['std'] = std

        index_lab = 0
        for i in self.dic_lab.keys():
            self.dic_lab[i]['index'] = index_lab
            index_lab += 1

        self.vital_length = len(list(self.dic_vital.keys()))
        self.lab_length = len(list(self.dic_lab.keys()))
        self.lab_list = self.dic_lab.keys()
        self.vital_list = self.dic_vital.keys()


    def assign_value_vital(self,hr_back,mrn_id):
        self.one_date_vital = np.zeros((self.time_sequence, self.vital_length))
        for i in range(self.time_sequence):
            self.hr_current = hr_back - self.time_sequence - self.predict_window + i
            if self.hr_crrent < 0:
                self.hr_current = 0

            self.one_data_vital[i,:] = self.assign_value_lab_single(self.hr_current,mrn_id)

    def assign_value_vital_single(self,hr_index,mrn_id):
        one_vital_sample = np.zeros(self.lab_length)
        if hr_index in self.dic_patient[mrn_id]['prior_time_vital']:
            for i in range(self.lab_length):
                lab_name = self.lab_list[i]
                if lab_name in self.dic_patient[mrn_id]['prior_time_vital'][hr_index]:
                    values = []
                    for k in self.dic_patient[mrn_id]['prior_time_vital'][hr_index][lab_name]:
                        try:
                            k_ = np.float(k)
                            values.append(k_)
                        except:
                            continue
                    value = np.mean(values)
                    mean_value = self.dic_lab[lab_name]['mean_value']
                    std_value = self.dic_lab[lab_name]['std']

                    one_vital_sample[i] = value


        return one_vital_sample




if __name__ == "__main__":
    read_d = read_data_covid()
