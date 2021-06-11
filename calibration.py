import numpy as np

class calibrate():
    def __init__(self, read_d, seq_cl):
        self.read_d = read_d
        self.seq_cl = seq_cl

    def fit(self, x, y):

        return self

    def decision_function(self, X):
        return  self.seq_cl.sess.run(self.seq_cl.logit_sig, feed_dict={self.seq_cl.input_x: X})