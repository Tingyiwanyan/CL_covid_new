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
        self.vital_length = len(list(self.dic_vital.keys()))
        self.lab_length = len(list(self.dic_lab.keys()))

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



if __name__ == "__main__":
    read_d = read_data_covid()
