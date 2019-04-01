from test.base import Policy, StochasticPolicy
from test.misc import ext

import numpy as np
import tensorflow as tf
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class MyModel(Model):
    def __init__(self,
                 name,
                 dim_output,
                 hidden_sizes,
                 hidden_W_init='glorot_uniform',
                 hidden_b_init='zeros',
                 output_W_init='glorot_uniform',
                 output_b_init='zeros',
                 hidden_nonlinearity='relu',
                 output_nonlinearity='linear',
                 weight_normalization=False):
        super().__init__()
        self.name = name
        self.dim_output = dim_output
        self.hidden_layers = list()
        for dim_h in hidden_sizes:
            self.hidden_layers.append(Dense(dim_h,
                                            kernel_initializer=hidden_W_init,
                                            bias_initializer=hidden_b_init,
                                            activation=hidden_nonlinearity))

        self.output_layer = Dense(dim_output,
                                  kernel_initializer=output_W_init,
                                  bias_initializer=output_b_init,
                                  activation=output_nonlinearity)

    def call(self, x):
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        x = self.output_layer(x)

        return x


class MAMLGaussianMLPPolicy(StochasticPolicy):
    def __init__(self,
                 name,
                 env_spec,
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity='relu',
                 output_nonlinearity='linear',
                 mean_network=None,
                 std_network=None,
                 std_parametrization='exp',
                 grad_step_size=1.0,
                 stop_grad=False,
                 ):
        # assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim, )
        self.step_size = grad_step_size
        self.stop_grad = stop_grad
        if type(self.step_size) == list:
            raise NotImplementedError

        if mean_network is None:
            self.mean_network = MyModel('mean_network',
                                        dim_output=self.action_dim,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=hidden_nonlinearity,
                                        output_nonlinearity=output_nonlinearity)

            # self.all_params = self.create_MLP(
            #     name='mean_network',
            #     output_dim=self.action_dim,
            #     hidden_sizes=hidden_sizes,
            # )
            # self.input_tensor, _ = self.forward_MLP('mean_network', self.all_params, reuse=None)
            # forward_mean = lambda x, params, is_train: self.forward_MLP('mean_network', params, input_tensor=x, is_training=is_train)[1]
        else:
            raise NotImplementedError

        if std_network is not None:
            raise NotImplementedError
        else:
            if adaptive_std:
                raise NotImplementedError
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError

                self.mean_network = MyModel('output_std_network',
                                            dim_output=self.action_dim,
                                            hidden_sizes=hidden_sizes,
                                            hidden_nonlinearity=hidden_nonlinearity,
                                            output_nonlinearity=output_nonlinearity)

                self.all_params['std_param'] = make_param_layer(
                    num_units=self.action_dim,
                    param=tf.constant_initializer(init_std_param),
                    name='output_std_param',
                    trainable=learn_std,
                )
                forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
            self.all_param_vals = None

            self._forward = lambda obs, params, is_train: (
                forward_mean(obs, params, is_train), forward_std(obs, params))

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._dist = DiagonalGaussian(self.action_dim)

            self._cached_params = {}

            super().__init__(env_spec)

            dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
            mean_var = dist_info_sym['mean']
            log_std_var = dist_info_sym['log_std']

            self._init_f_dist = tensor_utils.compile_function(
                inputs=[self.input_tensor],
                outputs=[mean_var, log_std_var],
            )
            self._cur_f_dist = self._init_f_dist

    def switch_to_init_dist(self):
        self._cur_f_dist = self._init_f_dist
        self._cur_f_dist_i = None
        self.all_params_vals = None


    def compute_updated_dists(self, samples):
        start = time.time()
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        sess = tf.get_default_session()

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i], 'observations', 'actions', 'advantages')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = obs_list + action_list + adv_list

        init_param_values = None









