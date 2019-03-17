
import tensorflow as tf
import time


class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError


class MyModel(RLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 baseline,
                 scope=None,
                 start_itr=0,
                 n_itr=100,
                 **kwargs):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.start_itr = start_itr
        self.n_itr = n_itr

    def train(self):


