import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import tensorflow as tf
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

class seq_cl():
    """
    create deep learning model
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
        self.len_death = len(self.death_data)
        self.len_live = len(self.live_data)


        self.batch_size = 64
        self.vital_length = self.read_d.vital_length
        self.lab_length = self.read_d.lab_length
        self.epoch = 10
        self.epoch_pre = 10
        self.gamma = 2
        self.tau = 1
        self.latent_dim = 100
        self.layer2_dim = 50
        self.layer3_dim = 32
        self.final_dim = self.layer2_dim
        self.boost_iteration = 10
        self.time_sequence = self.read_d.time_sequence
        self.positive_sample_size = 5
        self.negative_sample_size = self.batch_size


    def create_memory_bank(self,hr_onset):
        self.memory_bank_cohort = np.zeros((self.len_death,self.time_sequence,
                                         self.vital_length + self.lab_length))
        self.memory_bank_control = np.zeros((self.len_live,self.time_sequence,
                                         self.vital_length + self.lab_length))

        for i in range(self.len_death):
            name = self.death_data[i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            self.memory_bank_cohort[i, :, :] = one_data

        for i in range(self.len_live):
            name = self.live_data[i]
            self.read_d.return_tensor_data(name)
            one_data = self.read_d.one_data_tensor
            self.memory_bank_control[i, :, :] = one_data

    def construct_knn_feature_cohort(self,hr_onset):
        print("Im here in constructing feature graph")
        self.knn_sim_matrix = np.zeros((self.len_death,
                                        self.final_dim))
        self.knn_neighbor_feature = {}

        self.one_data_death = np.zeros((self.len_death,self.final_dim))
        for i in range(self.len_death):
            name = self.death_data[i]
            self.read_d.return_tensor_data_dynamic(name, hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_data_death[i,:] = one_data[0,:]
            #one_data = np.mean(one_data, 0)

        self.knn_sim_matrix = self.sess.run(self.one_data_death,
                                  feed_dict={self.input_x: self.one_batch_data})

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.len_death, algorithm='auto',metric='cosine').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.len_death):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            # center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.death_data[i]
            index = 0
            for j in range(self.len_death):
                if index == self.positive_sample_size:
                    break
                # compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i, j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor_feature.keys():
                    self.knn_neighbor_feature[center_patient_id_name] = {}
                    self.knn_neighbor_feature[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor_feature[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1

    def construct_knn_feature_control(self,hr_onset):
        print("Im here in constructing feature graph")
        self.knn_sim_matrix = np.zeros((self.len_live,
                                        self.final_dim))
        self.knn_neighbor_control_feature = {}

        self.one_data_live = np.zeros((self.len_live,self.final_dim))
        for i in range(self.len_live):
            name = self.live_data[i]
            self.read_d.return_tensor_data_dynamic(name, hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_data_death[i,:] = one_data[0,:]
            #one_data = np.mean(one_data, 0)

        self.knn_sim_matrix = self.sess.run(self.one_data_live,
                                  feed_dict={self.input_x: self.one_batch_data})

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.len_live, algorithm='auto',metric='cosine').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.len_live):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            # center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.live_data[i]
            index = 0
            for j in range(self.len_live):
                if index == self.positive_sample_size:
                    break
                # compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i, j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor_control_feature.keys():
                    self.knn_neighbor_control_feature[center_patient_id_name] = {}
                    self.knn_neighbor_control_feature[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor_control_feature[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1

    def construct_knn_attribute_cohort(self,hr_onset):
        """
        construct knn graph at every epoch using attribute information
        """
        print("Im here in constructing knn graph")

        self.knn_sim_matrix = np.zeros((self.len_death,
                                        self.vital_length + self.lab_length))
        self.knn_neighbor = {}

        for i in range(self.len_death):
            name = self.death_data[i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            one_data = np.mean(one_data, 0)
            self.knn_sim_matrix[i, :] = one_data

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.len_death, algorithm='auto', metric='euclidean').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.len_death):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            # center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.death_data[i]
            index = 0
            for j in range(self.len_death):
                if index == self.positive_sample_size:
                    break
                # compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i, j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor.keys():
                    self.knn_neighbor[center_patient_id_name] = {}
                    self.knn_neighbor[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1

    def construct_knn_attribute_control(self,hr_onset):
        """
        construct knn graph at every epoch using attribute information
        """
        print("Im here in constructing knn graph")

        self.knn_sim_matrix = np.zeros((self.len_live,
                                        self.vital_length + self.lab_length))
        self.knn_neighbor_control = {}

        for i in range(self.len_live):
            name = self.live_data[i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            one_data = np.mean(one_data, 0)
            self.knn_sim_matrix[i, :] = one_data

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.len_live, algorithm='auto', metric='euclidean').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.len_live):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            #center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.live_data[i]
            index = 0
            for j in range(self.len_live):
                if index == self.positive_sample_size:
                    break
                #compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i,j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor_control.keys():
                    self.knn_neighbor_control[center_patient_id_name] = {}
                    self.knn_neighbor_control[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor_control[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1


    def LSTM_layers(self):
        self.lstm = tf.keras.layers.LSTM(self.latent_dim,return_sequences=True,return_state=True)
        self.input_x = tf.keras.backend.placeholder(
            [None, self.time_sequence,self.vital_length + self.lab_length])
        self.whole_seq_output,self.final_memory_state,self.final_carry_state = self.lstm(self.input_x)
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])

        """
        positive sample
        """
        self.input_x_pos = tf.keras.backend.placeholder(
            [self.batch_size*self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length])
        self.whole_seq_output_pos, self.final_memory_state_pos, self.final_carry_state_pos = self.lstm(self.input_x_pos)
        self.whole_seq_out_pos_reshape = tf.reshape(self.whole_seq_output_pos,[self.batch_size,
                                                                               self.positive_sample_size,
                                                                               self.time_sequence,
                                                                               self.latent_dim])

        """
        negative sample
        """
        self.input_x_neg = tf.keras.backend.placeholder(
            [self.batch_size * self.negative_sample_size, self.time_sequence,
             self.vital_length + self.lab_length])
        self.whole_seq_output_neg, self.final_memory_state_neg, self.final_carry_state_neg = self.lstm(self.input_x_neg)
        self.whole_seq_out_neg_reshape = tf.reshape(self.whole_seq_output_neg, [self.batch_size,
                                                                                self.negative_sample_size,
                                                                                self.time_sequence,
                                                                                self.latent_dim])

    def LSTM_layers_stack(self, whole_seq_input, seq_input_pos, seq_input_neg, output_dim):
        lstm = tf.keras.layers.LSTM(output_dim, return_sequences=True, return_state=True)

        #whole_seq_output,final_memory_state,final_carry_state = lstm(whole_seq_input_act)
        dense = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                      activity_regularizer=tf.keras.regularizers.l2(0.01)
                                      )
        layer = tf.keras.layers.Dropout(.2, input_shape=(output_dim,))
        BN = tf.keras.layers.BatchNormalization()

        whole_seq_output_ = dense(whole_seq_input)
        whole_seq_output_bn = BN(whole_seq_output_)
        whole_seq_output_act = layer(whole_seq_output_bn)

        whole_seq_output, final_memory_state, final_carry_state = lstm(whole_seq_output_act)

        """
        positive sample
        """
        whole_seq_output_pos_ = dense(seq_input_pos)
        whole_seq_output_pos_bn = BN(whole_seq_output_pos_)
        whole_seq_output_pos_act = layer(whole_seq_output_pos_bn)
        whole_seq_output_pos, final_memory_state_pos, final_carry_state_pos = lstm(whole_seq_output_pos_act)

        """
        negative sample
        """
        whole_seq_output_neg_= dense(seq_input_neg)
        whole_seq_output_neg_bn = BN(whole_seq_output_neg_)
        whole_seq_output_neg_act = layer(whole_seq_output_neg_bn)
        whole_seq_output_neg, final_memory_state_neg, final_carry_state_neg = lstm(whole_seq_output_neg_act)

        return whole_seq_output, whole_seq_output_pos, whole_seq_output_neg


    def config_model(self):
        self.create_memory_bank(self.read_d.time_sequence)
        self.construct_knn_attribute_cohort(self.read_d.time_sequence)
        self.construct_knn_attribute_control(self.read_d.time_sequence)
        self.LSTM_layers()
        """
        LSTM stack layers
        """

        whole_seq_output,whole_seq_output_pos,whole_seq_output_neg = \
            self.LSTM_layers_stack(self.whole_seq_output,
                                   self.whole_seq_output_pos,self.whole_seq_output_neg,self.layer2_dim)
        #whole_seq_output, whole_seq_output_pos, whole_seq_output_neg = \
            #self.LSTM_layers_stack(whole_seq_output1,
                                   #whole_seq_output_pos1, whole_seq_output_neg1, self.layer3_dim)
        
        self.whole_seq_out_pos_reshape = tf.reshape(whole_seq_output_pos, [self.batch_size,
                                                                                self.positive_sample_size,
                                                                                self.time_sequence,
                                                                                self.final_dim])
        self.whole_seq_out_neg_reshape = tf.reshape(whole_seq_output_neg, [self.batch_size,
                                                                                self.negative_sample_size,
                                                                                self.time_sequence,
                                                                                self.final_dim])

        bce = tf.keras.losses.BinaryCrossentropy()
        self.x_origin = whole_seq_output[:,self.time_sequence-1,:]
        self.x_skip_contrast = self.whole_seq_out_pos_reshape[:,:,self.time_sequence-1,:]
        self.x_negative_contrast = self.whole_seq_out_neg_reshape[:,:,self.time_sequence-1,:]
        self.contrastive_learning()

        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                                   activity_regularizer=tf.keras.regularizers.l2(0.01),
                                                   activation=tf.nn.sigmoid)
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_cl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.log_normalized_prob)
        """
        focal loss
        """
        alpha = 0.25
        alpha_t = self.input_y_logit * alpha + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (1 - alpha)

        p_t = self.input_y_logit * self.logit_sig + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (
                tf.ones_like(self.input_y_logit) - self.logit_sig) + tf.keras.backend.epsilon()

        self.focal_loss_ = - alpha_t * tf.math.pow((tf.ones_like(self.input_y_logit) - p_t), self.gamma) * tf.math.log(
            p_t)
        self.focal_loss = tf.reduce_mean(self.focal_loss_)
        self.train_step_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.focal_loss)
        self.train_step_combine_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.focal_loss + 0.8 * self.log_normalized_prob)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def contrastive_learning(self):
        """
         positive inner product
         """
        self.x_origin_cl = tf.expand_dims(self.x_origin,axis=1)
        self.positive_broad = tf.broadcast_to(self.x_origin_cl,
                                              [self.batch_size, self.positive_sample_size, self.final_dim])
        self.negative_broad = tf.broadcast_to(self.x_origin_cl,
                                              [self.batch_size, self.negative_sample_size, self.final_dim])

        self.positive_broad_norm = tf.math.l2_normalize(self.positive_broad, axis=2)
        self.positive_sample_norm = tf.math.l2_normalize(self.x_skip_contrast, axis=2)

        self.positive_dot_prod = tf.multiply(self.positive_broad_norm, self.positive_sample_norm)
        self.positive_check_prod = tf.reduce_sum(self.positive_dot_prod, 2)
        # self.positive_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2)),1)
        self.positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2) / self.tau)

        """
        negative inner product
        """
        self.negative_broad_norm = tf.math.l2_normalize(self.negative_broad, axis=2)
        self.negative_sample_norm = tf.math.l2_normalize(self.x_negative_contrast, axis=2)

        self.negative_dot_prod = tf.multiply(self.negative_broad_norm, self.negative_sample_norm)
        self.negative_check_prod = tf.reduce_sum(self.negative_dot_prod, 2)
        self.negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.negative_dot_prod, 2) / self.tau), 1)
        self.negative_dot_prod_sum = tf.expand_dims(self.negative_dot_prod_sum, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer = tf.math.add(self.positive_dot_prod_sum, self.negative_dot_prod_sum)
        self.normalized_prob_log = tf.math.log(tf.math.divide(self.positive_dot_prod_sum, self.denominator_normalizer))
        self.normalized_prob_log_k = tf.reduce_sum(self.normalized_prob_log, 1)
        self.log_normalized_prob = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k, 0))


    def aquire_batch_data(self, starting_index, data_set,length,hr_onset):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length))
        self.one_batch_logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_batch_logit_dp[i, 0] = self.read_d.logit_label
            self.one_batch_data[i,:,:] = one_data
        self.one_batch_logit = list(self.one_batch_logit_dp[:,0])

    def aquire_batch_data_cl(self,starting_index, data_set,length,hr_onset):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length))
        self.one_batch_data_pos = np.zeros((length*self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length))
        self.one_batch_data_neg = np.zeros((length*self.negative_sample_size, self.time_sequence,
             self.vital_length + self.lab_length))

        #self.one_batch_logit = np.array(list(logit_input[starting_index:starting_index+length]))
        self.one_batch_logit_dp = np.zeros((length,1))
        #self.one_batch_logit_dp[:,0] = self.one_batch_logit
        for i in range(length):
            name = data_set[starting_index + i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_batch_logit_dp[i, 0] = self.read_d.logit_label
            label = self.read_d.logit_label
            self.one_batch_data[i, :, :] = one_data
            self.aquire_pos_data_random(label)
            self.aquire_neg_data_random(label)
            self.one_batch_data_pos[i*self.positive_sample_size:(i+1)*self.positive_sample_size,:,:] = \
                self.patient_pos_sample_tensor
            self.one_batch_data_neg[i*self.negative_sample_size:(i+1)*self.negative_sample_size,:,:] = \
                self.patient_neg_sample_tensor
        self.one_batch_logit = list(self.one_batch_logit_dp[:, 0])


    def aquire_batch_data_cl_attribute(self, starting_index, data_set, length,hr_onset):
        self.one_batch_data = np.zeros(
            (length, self.time_sequence, self.vital_length + self.lab_length))
        self.one_batch_data_pos = np.zeros((length * self.positive_sample_size, self.time_sequence,
                                            self.vital_length + self.lab_length))
        self.one_batch_data_neg = np.zeros((length * self.negative_sample_size, self.time_sequence,
                                            self.vital_length + self.lab_length))

        self.one_batch_logit_dp = np.zeros((length, 1))
        for i in range(length):
            name = data_set[starting_index + i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_batch_logit_dp[i, 0] = self.read_d.logit_label
            label = self.read_d.logit_label
            self.one_batch_data[i, :, :] = one_data
            self.aquire_pos_data_attribute(label,name)
            self.aquire_neg_data_random(label)
            self.one_batch_data_pos[i * self.positive_sample_size:(i + 1) * self.positive_sample_size, :, :] = \
                self.patient_pos_sample_tensor
            self.one_batch_data_neg[i * self.negative_sample_size:(i + 1) * self.negative_sample_size, :, :] = \
                self.patient_neg_sample_tensor

    def aquire_batch_data_cl_feature(self, starting_index, data_set, length,hr_onset):
        self.one_batch_data = np.zeros(
            (length, self.time_sequence, self.vital_length + self.lab_length))
        self.one_batch_data_pos = np.zeros((length * self.positive_sample_size, self.time_sequence,
                                            self.vital_length + self.lab_length))
        self.one_batch_data_neg = np.zeros((length * self.negative_sample_size, self.time_sequence,
                                            self.vital_length + self.lab_length))

        self.one_batch_logit_dp = np.zeros((length, 1))
        for i in range(length):
            name = data_set[starting_index + i]
            self.read_d.return_tensor_data_dynamic(name,hr_onset)
            one_data = self.read_d.one_data_tensor
            self.one_batch_logit_dp[i, 0] = self.read_d.logit_label
            label = self.read_d.logit_label
            self.one_batch_data[i, :, :] = one_data
            self.aquire_pos_data_feature(label,name)
            self.aquire_neg_data_random(label)
            self.one_batch_data_pos[i * self.positive_sample_size:(i + 1) * self.positive_sample_size, :, :] = \
                self.patient_pos_sample_tensor
            self.one_batch_data_neg[i * self.negative_sample_size:(i + 1) * self.negative_sample_size, :, :] = \
                self.patient_neg_sample_tensor

    def aquire_pos_data_random(self,label):
        #print("im in pos")
        self.patient_pos_sample_tensor = \
            np.zeros((self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length))
        if label == 1:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.len_death, self.positive_sample_size)).astype(int)
            self.patient_pos_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
        else:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.len_live, self.positive_sample_size)).astype(int)
            self.patient_pos_sample_tensor = self.memory_bank_control[index_neighbor, :, :]


    def aquire_neg_data_random(self,label):
        #print("im in neg")
        self.patient_neg_sample_tensor = \
            np.zeros((self.negative_sample_size, self.time_sequence,
                      self.vital_length + self.lab_length))
        if label == 1:
            #index_neighbor = \
                #np.floor(np.random.uniform(0, self.len_live, self.negative_sample_size)).astype(int)
            #self.patient_neg_sample_tensor = self.memory_bank_control[index_neighbor,:,:]
            self.patient_neg_sample_tensor = self.memory_bank_control[0:self.negative_sample_size, :, :]
        else:
            #index_neighbor = \
                #np.floor(np.random.uniform(0, self.len_death, self.negative_sample_size)).astype(int)
            #self.patient_neg_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
            self.patient_neg_sample_tensor = self.memory_bank_cohort[0:self.negative_sample_size, :, :]

    def aquire_pos_data_feature(self,label,name):
        #print("im in pos")
        self.patient_pos_sample_tensor = \
            np.zeros((self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length))
        if label == 1:
            index_neighbor = np.array(self.knn_neighbor_feature[name]['knn_neighbor'])
            self.patient_pos_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
        else:
            index_neighbor =  np.array(self.knn_neighbor_control_feature[name]['knn_neighbor'])
            self.patient_pos_sample_tensor = self.memory_bank_control[index_neighbor, :, :]

    def aquire_pos_data_attribute(self,label,name):
        #print("im in pos")
        self.patient_pos_sample_tensor = \
            np.zeros((self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length))
        if label == 1:
            index_neighbor = np.array(self.knn_neighbor[name]['knn_neighbor'])
            self.patient_pos_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
        else:
            index_neighbor =  np.array(self.knn_neighbor_control[name]['knn_neighbor'])
            self.patient_pos_sample_tensor = self.memory_bank_control[index_neighbor, :, :]

    def pre_train(self):
        self.iteration = np.int(np.floor(np.float(self.len_train) / self.batch_size))
        for i in range(self.epoch_pre):
            for j in range(self.iteration):
                #print(j)
                self.aquire_batch_data_cl_attribute(j*self.batch_size, self.train_data, self.batch_size,self.read_d.time_sequence)
                self.err_ = self.sess.run([self.log_normalized_prob, self.train_step_cl,self.logit_sig],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_y_logit: self.one_batch_logit_dp,
                                                     self.input_x_pos:self.one_batch_data_pos,
                                                     self.input_x_neg:self.one_batch_data_neg})

                #print(self.err_[0])
                #print(roc_auc_score(self.one_batch_logit, self.err_[2]))
            print("epoch")
            print(i)
            self.test()

    def train(self):
        self.step = []
        self.acc = []
        self.iteration = np.int(np.floor(np.float(self.len_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                #print(j)
                self.aquire_batch_data_cl(j*self.batch_size, self.train_data, self.batch_size,self.read_d.time_sequence)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_combine_fl,self.logit_sig],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_y_logit: self.one_batch_logit_dp,#})
                                                     self.input_x_pos:self.one_batch_data_pos,
                                                     self.input_x_neg:self.one_batch_data_neg})
            print("epoch")
            print(i)
            self.val()

            #print(self.err_[0])
            #auc = roc_auc_score(self.one_batch_logit, self.err_[2])
            #print(auc)
            #self.step.append(j)
            #self.acc.append(auc)

            #self.test()

    def test(self):
        sample_size = np.int(np.floor(self.len_test * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            test = resample(self.test_data, n_samples=sample_size)
            self.aquire_batch_data(0, test, len(test),self.read_d.time_sequence)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
            auc.append(roc_auc_score(self.one_batch_logit, self.out_logit))
            auprc.append(
                average_precision_score(self.one_batch_logit, self.out_logit))
        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))


    def val(self):
        self.aquire_batch_data(0, self.validate_data, self.len_validate,self.read_d.time_sequence)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print("auc")
        print(roc_auc_score(self.one_batch_logit, self.out_logit))
        print("auprc")
        print(average_precision_score(self.one_batch_logit, self.out_logit))

    def real_time_prediction(self,name):
        self.hour = []
        self.mortality_risk = []
        if self.read_d.dic_patient[name]['death_flag'] == 1:
            self.logit_label = 1
            self.hr_onset = np.float(self.read_d.dic_patient[name]['death_hour'])
        else:
            self.logit_label = 0
            prior_times = np.max([np.float(i) for i in self.read_d.dic_patient[name]['prior_time_vital']])
            #self.hr_onset = np.floor(np.random.uniform(0, hr_onset_up, 1))
            self.hr_onset = prior_times

        for i in range(int(self.hr_onset-self.read_d.predict_window)):
            self.one_data_tensor = np.zeros((1,self.time_sequence, self.vital_length + self.lab_length))
            self.predict_window_start = i
            self.read_d.assign_value_vital(self.predict_window_start, name)
            self.one_data_tensor[0,:, 0:self.vital_length] = self.read_d.one_data_vital
            self.read_d.assign_value_lab(self.predict_window_start, name)
            self.one_data_tensor[0,:, self.vital_length:self.vital_length + self.lab_length] = self.read_d.one_data_lab
            self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_data_tensor})[0][0]
            self.hour.append(i)
            self.mortality_risk.append(self.out_logit)

