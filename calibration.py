import numpy as np

class calibrate():
    def __init__(self, read_d, seq_cl):
        self.read_d = read_d
        all_data = list(self.read_d.dic_patient.keys())
        self.test_data = all_data[3891:]
        self.len_test = len(self.test_data)
        self.seq_cl = seq_cl
        self.input = np.zeros((3, 4))
        self.y = [0,1,0]

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        self.seq_cl.aquire_batch_data(0, self.test_data, self.len_test, self.read_d.time_sequence)
        return  self.seq_cl.sess.run(self.seq_cl.logit_sig, feed_dict={self.seq_cl.input_x: self.seq_cl.one_batch_data})[:,0]