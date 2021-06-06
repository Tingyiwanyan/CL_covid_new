import numpy as np
import random
import math
import time
import pandas as pd
import json


def __init__(self):
    file_path = '/datadrive/tingyi_wanyan/user_tingyi.wanyan/tensorflow_venv/registry_2020-06-29'
    self.reg = file_path + '/registry.csv'
    self.covid_lab = file_path + '/covid19LabTest.csv'
    self.lab = file_path + '/Lab.csv'
    self.vital = file_path + '/vitals.csv'
    file_path_ = '/home/tingyi.wanyan'
    self.lab_comb = 'lab_mapping_comb.csv'
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