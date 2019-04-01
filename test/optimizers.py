from test.misc import ext

import tensorflow as tf
import scipy.optimize
import time

class LbfgsOptimizer:
    def __init__(self, name, max_opt_itr=20, callback=None):
        self._name = name
        self._max_opt_itr =max_opt_itr
        self._opt_fun = None
        self._target = None
        self._callback = callback

    def optimize(self, inputs, extra_inputs=None):
        f_opt = self._opt_fun['f_opt']