
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

    def obtain_samples(self, itr):


    def train(self):
        with tf.Session() as sess:
            self.init_opt()
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.initialize_variables(uninit_vars))
            start_time = time.time()
            env = self.env

    def init_opt(self):


        raise NotImplementedError



