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

        for i in self.dic_lab.keys():
            values = []
            for k in self.dic_lab[i]['lab_value_patient']:
                try:
                    k_ = np.float(k)
                    values.append(k_)
                except:
                    continue
            up_value = np.percentile(values,95)
            low_value = np.percentile(values,5)
            values_ = [k for k in values if not (values>up_value and value<low_value)]
            mean_lab = np.mean(values_)
            std_lab = np.std(values_)
            kg.dic_lab[i]['mean_value'] = mean_lab
            kg.dic_lab[i]['std'] = std_lab

        
        for i in self.dic_vital.keys():
            values = []
            for k in self.dic_vital[i]['lab_value_patient']:
                try:
                    k_ = np.float(k)
                    values.append(k_)
                except:
                    continue
            up_value = np.percentile(values, 95)
            low_value = np.percentile(values, 5)
            values_ = [k for k in values if not (values > up_value and value < low_value)]
            mean = np.mean(values_)
            std = np.std(values_)
            kg.dic_vital[i]['mean_value'] = mean
            kg.dic_vital[i]['std'] = std



if __name__ == "__main__":
    read_d = read_data_covid()
