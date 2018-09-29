import numpy as np
import tensorflow as tf


class HMM(object):
    def __init__(self, initial_prob, trans_prob, obs_prob):
        self.N = np.size(initial_prob)
        self.initial_prob = initial_prob
        self.trans_prob = trans_prob
        self.obs_prob = obs_prob
        self.emission = tf.constant(obs_prob)
        assert self.initial_prob.shape == (self.N, 1)
        assert self.trans_prob.shape == (self.N, self.N)
        assert self.obs_prob.shape[0] == self.N
        self.obs = tf.placeholder(tf.int32)
        self.fwd = tf.placeholder(tf.float64)
        self.viterbi = tf.placeholder(tf.float64)

    def get_emission(self, obs_idx):
        slice_location = [0, obs_idx]
        num_rows = tf.shape(self.emission)[0]
        slice_shape = [num_rows, 1]
        return tf.slice(self.emission, slice_location, slice_shape)

    def forward_init_op(self):
        obs_prob = self.get_emission(self.obs)
        fwd = tf.multiply(self.initial_prob, obs_prob)
        return fwd

    def forward_op(self):
        transitions = tf.matmul(self.fwd, tf.transpose(self.get_emission(self.obs)))
        weighted_transitions = transitions * self.trans_prob
        fwd = tf.reduce_sum(weighted_transitions, 0)
        return tf.reshape(fwd, tf.shape(self.fwd))

    def decode_op(self):
        transitions = tf.matmul(self.viterbi, tf.transpose(self.get_emission(self.obs)))
        weighted_transitions = transitions * self.trans_prob
        viterbi = tf.reduce_max(weighted_transitions, 0)
        return tf.reshape(viterbi, tf.shape(self.viterbi))

    def backpt_op(self):
        back_transitions = tf.matmul(self.viterbi, np.ones((1, self.N)))
        weighted_back_transitions = back_transitions * self.trans_prob
        return tf.argmax(weighted_back_transitions, 0)