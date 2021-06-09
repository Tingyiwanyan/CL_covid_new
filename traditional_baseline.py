import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np

class tradition_b():
    """
    Create tradition model
    """

    def __init__(self, read_d):
        self.read_d = read_d
        all_data = list(self.read_d.dic_patient.keys())
        self.train_data = all_data[0:3405]
        self.validate_data = all_data[3405:3891]
        self.test_data = all_data[3891:]
        self.death_data = self.read_d.death_data
        self.live_data = self.read_d.live_data
        self.len_train = len(self.train_data)
        self.len_validate = len(self.validate_data)
        self.len_test = len(self.test_data)

        self.vital_length = self.read_d.vital_length
        self.lab_length = self.read_d.lab_length
        self.latent_dim = 100
        self.epoch = 20
        self.gamma = 2
        self.tau = 1
        self.batch_size = 64

        self.lr = LogisticRegression(random_state=0)
        self.rf = RandomForestClassifier(max_depth=100,random_state=0)

    def aquire_batch_data(self, starting_index, data_set,length):
        self.one_batch_data = np.zeros((length,self.vital_length+self.lab_length))#+self.static_length))
        self.one_batch_logit = list(np.zeros(length))
        self.one_batch_logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.return_tensor_data(name)
            one_data = self.read_d.one_data_tensor
            #one_data[one_data==0]=np.nan
            #one_data = np.nan_to_num(np.nanmean(one_data,0))
            one_data = np.mean(one_data,0)
            self.one_batch_data[i,:] = one_data
            self.one_batch_logit[i] = self.read_d.logit_label
            self.one_batch_logit_dp[i,0] = self.read_d.logit_label
            #self.one_batch_data[i,self.vital_length+self.lab_length:] = self.read_d.one_data_tensor_static
            #self.one_batch_logit[i] = self.read_d.logit_label
            #self.one_batch_logit_dp[i,0] = self.read_d.logit_label


    def MLP_config(self):
        self.input_y_logit = tf.keras.backend.placeholder(
            [None, 1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.vital_length + self.lab_length])
        self.embedding = tf.compat.v1.layers.dense(inputs=self.input_x,
                                                   units=self.latent_dim,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.relu)
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.embedding,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)

        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

        alpha = 0.25
        alpha_t = self.input_y_logit * alpha + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (1 - alpha)

        p_t = self.input_y_logit * self.logit_sig + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (
                tf.ones_like(self.input_y_logit) - self.logit_sig) + tf.keras.backend.epsilon()

        self.focal_loss_ = - alpha_t * tf.math.pow((tf.ones_like(self.input_y_logit) - p_t), self.gamma) * tf.math.log(
            p_t)
        self.focal_loss = tf.reduce_mean(self.focal_loss_)
        self.train_step_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.focal_loss)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def MLP_train(self):
        #init_hidden_state = np.zeros(
            #(self.batch_size, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.iteration = np.int(np.floor(np.float(self.len_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                #print(j)
                self.aquire_batch_data(j*self.batch_size, self.train_data, self.batch_size)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     #self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit_dp})
                                                     #self.init_hiddenstate: init_hidden_state})
                #print(self.err_[0])
            print("epoch")
            print(i)
            self.MLP_val()

    def MLP_test(self):
        #init_hidden_state = np.zeros(
            #(self.length_test, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.aquire_batch_data(0, self.test_data, self.len_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print("auc")
        print(roc_auc_score(self.one_batch_logit, self.out_logit))
        print("auprc")
        print(average_precision_score(self.one_batch_logit, self.out_logit))

    def MLP_val(self):
        self.aquire_batch_data(0, self.validate_data, self.len_validate)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print("auc")
        print(roc_auc_score(self.one_batch_logit, self.out_logit))
        print("auprc")
        print(average_precision_score(self.one_batch_logit, self.out_logit))



    def logistic_regression(self):
        #self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        #for i in range(self.epoch):
            #for j in range(self.iteration):
                #print(j)
        self.aquire_batch_data(0,self.train_data,len(self.train_data))
        self.lr.fit(self.one_batch_data,self.one_batch_logit)
                #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                #print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

        self.test_logistic_regression()

    def test_logistic_regression(self):
        #self.aquire_batch_data(0,self.test_data,self.length_test)
        self.aquire_batch_data(0, self.test_data, len(self.test_data))
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(roc_auc_score(self.one_batch_logit, self.lr.predict_proba(self.one_batch_data)[:,1]))
        print("auprc")
        print(average_precision_score(self.one_batch_logit, self.lr.predict_proba(self.one_batch_data)[:, 1]))


    def random_forest(self):
        self.aquire_batch_data(0,self.train_data,len(self.train_data))
        self.rf.fit(self.one_batch_data, self.one_batch_logit)

        self.test_random_forest()

    def test_random_forest(self):
        self.aquire_batch_data(0, self.test_data, len(self.test_data))
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(roc_auc_score(self.one_batch_logit, self.rf.predict(self.one_batch_data)))
        print("auprc")
        print(average_precision_score(self.one_batch_logit, self.rf.predict(self.one_batch_data)))

    def real_time_prediction(self,name):
        self.hour = []
        self.mortality_risk = []
        if self.dic_patient[name]['death_flag'] == 1:
            self.logit_label = 1
            self.hr_onset = np.float(self.dic_patient[name]['death_hour'])
        else:
            self.logit_label = 0
            prior_times = np.max([np.float(i) for i in self.dic_patient[name]['prior_time_vital']])
            #self.hr_onset = np.floor(np.random.uniform(0, hr_onset_up, 1))
            self.hr_onset = np.floor(prior_times/2)

        self.predict_window_start = self.hr_onset-self.predict_window