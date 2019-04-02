import tensorflow as tf
import numpy as np

def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))
