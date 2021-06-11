import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from seq_cl import seq_cl
from traditional_baseline import tradition_b
from calibration import calibrate


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

        self.time_sequence = 6
        self.predict_window = 0

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

        name_list = list(self.dic_patient.keys())
        for i in name_list:
            if self.dic_patient[i]['prior_time_vital'] == {} or len(list(self.dic_patient[i]['prior_time_vital'].keys())) < 3:
                self.dic_patient.pop(i)
            #if len(list(self.dic_patient[i]['prior_time_vital'].keys())) < 3:
                #self.dic_patient.pop(i)

        name_list = list(self.dic_patient.keys())
        self.death_data = []
        self.live_data = []
        for i in name_list:
            if self.dic_patient[i]['death_flag'] == 1:
                self.death_data.append(i)
            else:
                self.live_data.append(i)

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
        self.lab_list = list(self.dic_lab.keys())
        self.vital_list = list(self.dic_vital.keys())

    def return_tensor_data_dynamic(self,name,hr_onset):
        self.one_data_tensor = np.zeros((self.time_sequence, self.vital_length + self.lab_length))
        if self.dic_patient[name]['death_flag'] == 1:
            self.logit_label = 1
        else:
            self.logit_label = 0

        self.predict_window_start = hr_onset - self.predict_window

        self.assign_value_vital(self.predict_window_start, name)
        self.one_data_tensor[:, 0:self.vital_length] = self.one_data_vital
        self.assign_value_lab(self.predict_window_start, name)
        self.one_data_tensor[:, self.vital_length:self.vital_length + self.lab_length] = self.one_data_lab

    def return_tensor_data(self,name):
        self.one_data_tensor = np.zeros((self.time_sequence,self.vital_length+self.lab_length))
        if self.dic_patient[name]['death_flag'] == 1:
            self.logit_label = 1
            self.hr_onset = np.float(self.dic_patient[name]['death_hour'])
        else:
            self.logit_label = 0
            prior_times = np.max([np.float(i) for i in self.dic_patient[name]['prior_time_vital']])
            #self.hr_onset = np.floor(np.random.uniform(0, hr_onset_up, 1))
            self.hr_onset = np.floor(prior_times/2)

        self.predict_window_start = self.hr_onset-self.predict_window

        self.assign_value_vital(self.predict_window_start,name)
        self.one_data_tensor[:,0:self.vital_length] = self.one_data_vital
        self.assign_value_lab(self.predict_window_start,name)
        self.one_data_tensor[:,self.vital_length:self.vital_length+self.lab_length] = self.one_data_lab



    def assign_value_vital(self,hr_back,mrn_id):
        self.one_data_vital = np.zeros((self.time_sequence, self.vital_length))
        for i in range(self.time_sequence):
            self.hr_current = np.float(hr_back - self.time_sequence + i)
            if self.hr_current < 0:
                self.hr_current = 0.0

            self.one_data_vital[i,:] = self.assign_value_vital_single(self.hr_current,mrn_id)

    def assign_value_vital_single(self,hr_index,mrn_id):
        one_vital_sample = np.zeros(self.vital_length)
        #prior_times = [np.float(i) for i in self.dic_patient[mrn_id]['prior_time_vital']]
        hr_index = str(hr_index)
        if hr_index in self.dic_patient[mrn_id]['prior_time_vital'].keys():
            for i in range(self.vital_length):
                vital_name = self.vital_list[i]
                if vital_name in self.dic_patient[mrn_id]['prior_time_vital'][hr_index]:
                    values = [np.float(k) for k in self.dic_patient[mrn_id]['prior_time_vital'][hr_index][vital_name]]
                    value = np.mean(values)
                    mean_value = self.dic_vital[vital_name]['mean_value']
                    std_value = self.dic_vital[vital_name]['std']

                    norm_value = (np.float(value) - mean_value) / std_value
                    if np.isnan(norm_value):
                        norm_value = 0
                    one_vital_sample[i] = norm_value


        return one_vital_sample

    def assign_value_lab(self,hr_back,mrn_id):
        self.one_data_lab = np.zeros((self.time_sequence, self.lab_length))
        for i in range(self.time_sequence):
            self.hr_current = np.float(hr_back - self.time_sequence + i)
            if self.hr_current < 0:
                self.hr_current = 0.0

            self.one_data_lab[i,:] = self.assign_value_lab_single(self.hr_current,mrn_id)

    def assign_value_lab_single(self,hr_index,mrn_id):
        one_lab_sample = np.zeros(self.lab_length)
        #prior_times = [np.float(i) for i in self.dic_patient[mrn_id]['prior_time_lab']]
        hr_index =str(hr_index)
        if hr_index in self.dic_patient[mrn_id]['prior_time_lab'].keys():
            for i in range(self.lab_length):
                lab_name = self.lab_list[i]
                if lab_name in self.dic_patient[mrn_id]['prior_time_lab'][hr_index]:
                    values = []
                    for k in self.dic_patient[mrn_id]['prior_time_lab'][hr_index][lab_name]:
                        try:
                            k_ = np.float(k)
                            values.append(k_)
                        except:
                            continue
                    value = np.mean(values)
                    mean_value = self.dic_lab[lab_name]['mean_value']
                    std_value = self.dic_lab[lab_name]['std']

                    norm_value = (np.float(value) - mean_value) / std_value
                    if np.isnan(norm_value):
                        norm_value = 0
                    one_lab_sample[i] = norm_value


        return one_lab_sample




if __name__ == "__main__":
    read_d = read_data_covid()
    seq = seq_cl(read_d)
    tb = tradition_b(read_d)
    #cal = calibrate(read_d,seq)
