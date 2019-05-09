
from test_rl.distribution import DiagonalGaussian
from v2.environment import PortfolioEnv

from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
import random
import tensorflow as tf
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, Reshape, BatchNormalization, ReLU


EP_MAX = 1000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01
LR = 0.0001
META_LR = 0.001
# BATCH = 8192
BATCH = 512
MINIBATCH = 32
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
SIGMA_FLOOR = 0.0



def discount(x, gamma, terminal_array=None):
    if terminal_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]


class RunningStats:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count


class FeatureNetwork(Model):
    def __init__(self):
        super(FeatureNetwork, self).__init__()
        # self.dim_input = list(env.observation_space.shape)
        # self.dim_output = env.action_space.shape[0]
        # self.reshape = Reshape(self.dim_input + [1])

        self.relu = ReLU()
        self.flatten = Flatten()
        self.batch_norm = BatchNormalization()

        self.conv1 = Conv2D(8, (1, 1), dtype=tf.float32)
        self.conv20 = Conv2D(8, (20, 1), dtype=tf.float32, padding='same')
        self.conv60 = Conv2D(8, (60, 1), dtype=tf.float32, padding='same')

        self.conv = Conv2D(4, (20, 1), strides=(20, 1), dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, 0)

        positional_encoding = np.zeros([1, x.shape[1], x.shape[2], 1])
        for pos in range(x.shape[1]):
            for i in range(x.shape[2]):
                i_ = i // 2
                if i % 2 == 0:
                    positional_encoding[:, pos, 2 * i_, :] = np.sin(pos / 10000 ** (2 * i_ / x.shape[2]))
                else:
                    positional_encoding[:, pos, 2 * i_ + 1, :] = np.cos(pos / 10000 ** (2 * i_ / x.shape[2]))

        x = x + tf.cast(positional_encoding, tf.float32)
        x_1 = self.conv1(x)
        x_1 = self.batch_norm(x_1)
        x_20 = self.conv20(x)
        x_20 = self.batch_norm(x_20)
        x_60 = self.conv60(x)
        x_60 = self.batch_norm(x_60)

        x = x_1 + x_20 + x_60
        x = self.relu(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.flatten(x)


class ParamNetwork(Model):
    def __init__(self, a_dim, dim_hidden=[64, 32, 16]):
        super(ParamNetwork, self).__init__()
        self.feature_net = FeatureNetwork()
        self.dim_hidden = dim_hidden

        self.hidden_layer = dict()
        for i, dim_h in enumerate(dim_hidden):
            self.hidden_layer['h' + str(i + 1)] = Dense(dim_h, activation='relu')

        self.output_layer_actor = Dense(a_dim, activation='sigmoid')
        self.output_layer_critic = Dense(1, activation='linear')

    @property
    def dim_input(self):
        return self.feature_net.dim_input

    @property
    def dim_output(self):
        return self.feature_net.dim_output

    def call(self, x):
        x = self.feature_net(x)
        for i in range(len(self.dim_hidden)):
            x = self.hidden_layer['h' + str(i + 1)](x)
        x_a = self.output_layer_actor(x)
        x_c = self.output_layer_critic(x)
        return x_a, x_c
        # return x_a / tf.reduce_sum(x_a, axis=1, keepdims=True), x_c


class MetaPPO:
    def __init__(self, env, M=256, K=128):
        self.discrete = False
        self.s_dim = env.observation_space.shape
        if len(env.observation_space.shape) > 0:
            self.a_dim = env.action_space.shape[0]
            self.a_bound = (env.action_space.high - env.action_space.low) / 2.
            self.a_min = env.action_space.low
            self.a_max = env.action_space.high
        else:
            self.a_dim = env.action_space.n

        self.dist = DiagonalGaussian(self.a_dim)

        self.param_network = ParamNetwork(self.a_dim)
        self.log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.old_param_network = ParamNetwork(self.a_dim)
        self.old_log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.meta_optimizer = tf.optimizers.Adam(META_LR)
        self.optimizer = tf.optimizers.Adam(LR)

        self.sampler = Sampler(env)

        self.M = M
        self.K = K

        self.global_step = 0
        self._initialize(env.reset())

        self.optim_param_net_wgt = self.param_network.get_weights()
        self.optim_log_sigma_wgt = self.log_sigma.numpy()

    def _initialize(self, s):
        _ = self.param_network(s)
        _ = self.old_param_network(s)

    def assign_old_network(self):
        self.old_param_network.set_weights(self.param_network.get_weights())
        self.old_log_sigma.assign(self.log_sigma.numpy())

    def evaluate_state(self, state, stochastic=True):
        mu, value_ = self.param_network(state)
        if stochastic:
            action = self.dist.sample({'mean': mu, 'log_std': self.log_sigma})
        else:
            action = mu
        return self.action_to_weight_zero(action), value_

    def action_to_weight_linear(self, action):
        a_positive = action - tf.math.reduce_min(action) + self.a_min
        return a_positive / tf.reduce_sum(a_positive, axis=1, keepdims=True)

    def action_to_weight_zero(self, action):
        a_positive = tf.clip_by_value(action, self.a_min, self.a_max)
        return a_positive / tf.reduce_sum(a_positive, axis=1, keepdims=True)

    def polynomial_epsilon_decay(self, learning_rate, global_step, decay_steps, end_learning_rate, power):
        global_step_ = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step_ / decay_steps) ** power \
                                + end_learning_rate

        return decayed_learning_rate

    def grad_loss(self, trajectory):
        s_batch, a_batch, r_batch, adv_batch = trajectory
        idx = np.arange(len(s_batch))
        np.random.shuffle(idx)

        batch_loss = 0
        batch_count = len(s_batch) // MINIBATCH
        for i in range(batch_count):
            epsilon_decay = self.polynomial_epsilon_decay(0.1, self.global_step, 1e5, 0.01, power=0.0)
            s_mini = s_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
            a_mini = a_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
            r_mini = r_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
            adv_mini = adv_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
            with tf.GradientTape() as tape:
                mu_old, v_old = self.old_param_network(s_mini)
                mu, v = self.param_network(s_mini)
                ratio = self.dist.likelihood_ratio_sym(
                    a_mini,
                    {'mean': mu_old * self.a_bound, 'log_std': self.old_log_sigma},
                    {'mean': mu * self.a_bound, 'log_std': self.log_sigma})
                # ratio = tf.maximum(logli, 1e-6) / tf.maximum(old_logli, 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = adv_mini.squeeze() * ratio
                surr2 = adv_mini.squeeze() * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                loss_pi = - tf.reduce_mean(tf.minimum(surr1, surr2))

                clipped_value_estimate = v_old + tf.clip_by_value(v - v_old, -epsilon_decay, epsilon_decay)
                loss_v1 = tf.math.squared_difference(clipped_value_estimate, r_mini)
                loss_v2 = tf.math.squared_difference(v, r_mini)
                loss_v = tf.reduce_mean(tf.maximum(loss_v1, loss_v2)) * 0.5

                entropy = self.dist.entropy({'mean': mu, 'log_std': self.log_sigma})
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

                loss_sum = tf.reduce_mean(tf.maximum(tf.reduce_sum(a_mini, axis=1) - 1, 0) + tf.maximum(
                    0.99 - tf.reduce_sum(a_mini, axis=1), 0)) * 100

                loss = loss_pi + loss_v * VF_COEFF + pol_entpen + loss_sum

            minigrad = tape.gradient(loss, self.param_network.trainable_variables + [self.log_sigma])
            if i == 0:
                grad = [minigrad[j] / batch_count for j in range(len(minigrad))]
            else:
                grad = [grad[j] + minigrad[j] / batch_count for j in range(len(minigrad))]

            batch_loss += loss.numpy() / batch_count

        print('loss: {} (last: loss_pi:{:.4f}/loss_v:{:.4f}/pol_entpen:{:.4f}/loss_sum:{:.4f}'.format(
            batch_loss, loss_pi, loss_v, pol_entpen, loss_sum))
        print('grad: {}'.format(grad[-1]))
        print('log sigma: {}'.format(self.log_sigma))

        return grad

    def metatrain(self, env_samples):
        self.assign_old_network()

        meta_grad = None

        for env_i, env_t in enumerate(env_samples):
            self.fast_train(env_t)
            trajectory_target = self.sampler.get_trajectories(self, env_t, env_t + self.K)

            self.param_network.set_weights(self.optim_param_net_wgt)
            self.log_sigma.assign(self.optim_log_sigma_wgt)

            grad = self.grad_loss(trajectory_target)
            print("env_i: {} / env_t: {}".format(env_i, env_t))
            if meta_grad is None:
                meta_grad = [grad[k] / len(env_samples) for k in range(len(grad))]
            else:
                meta_grad = [meta_grad[k] + grad[k] / len(env_samples) for k in range(len(grad))]

        self.meta_optimizer.apply_gradients(zip(meta_grad, self.param_network.trainable_variables + [self.log_sigma]))

        self.optim_param_net_wgt = self.param_network.get_weights()
        self.optim_log_sigma_wgt = self.log_sigma.numpy()

    def fast_train(self, env_t):
        self.param_network.set_weights(self.optim_param_net_wgt)
        self.log_sigma.assign(self.optim_log_sigma_wgt)

        trajectory_sample = self.sampler.get_trajectories(self, env_t - self.M, env_t, render=False)
        grad = self.grad_loss(trajectory_sample)
        self.optimizer.apply_gradients(zip(grad, self.param_network.trainable_variables + [self.log_sigma]))

    def save_model(self, f_name):
        w_dict = {}
        # w_dict['param_network'] = self.param_network.get_weights()
        # w_dict['log_sigma'] = self.log_sigma.numpy()
        w_dict['param_network'] = self.optim_param_net_wgt
        w_dict['log_sigma'] = self.optim_log_sigma_wgt
        w_dict['global_step'] = self.global_step

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)

        self.optim_param_net_wgt = w_dict['param_network']
        self.optim_log_sigma_wgt = w_dict['log_sigma']
        self.param_network.set_weights(self.optim_param_net_wgt)
        self.log_sigma.assign(self.optim_log_sigma_wgt)
        self.global_step = w_dict['global_step']

        print("model loaded. (path: {})".format(f_name))


class Sampler:
    def __init__(self, env):
        self.env = env

    def get_trajectories(self, model, begin_t=None, end_t=None, render=False, stochastic=True, statistics=False):
        buffer_s, buffer_a, buffer_v, buffer_r, buffer_done = [], [], [], [], []
        rolling_r = RunningStats()

        s = self.env.reset(begin_t)
        done = False
        t_step = 0

        batch_size = end_t - begin_t
        while len(buffer_r) < batch_size:
            a, v = model.evaluate_state(s, stochastic=stochastic)

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(tf.squeeze(v))
            buffer_done.append(done)

            # a = tf.clip_by_value(a, self.env.action_space.low, self.env.action_space.high)
            s, r, _, done = self.env.step(np.squeeze(a))
            buffer_r.append(r)
            t_step += 1

            # print(t_step)
            if done:
                s = self.env.reset(begin_t)
                done = False
                t_step = 0

        rewards = np.array(buffer_r)
        rolling_r.update(rewards)
        rewards = np.clip(rewards / rolling_r.std, -10, 10)

        v_final = [tf.squeeze(v) * (1 - done)]
        values = np.array(buffer_v + v_final)
        dones = np.array(buffer_done + [done])

        # Generalized Advantage Estimation
        delta = rewards + GAMMA * values[1:] * (1 - dones[1:]) - values[:-1]
        adv = discount(delta, GAMMA * LAMBDA, dones)
        returns = adv + np.array(buffer_v)
        adv = (adv - adv.mean()) / np.maximum(adv.std(), 1e-6)

        s_batch, a_batch, r_batch, adv_batch = np.reshape(buffer_s, (batch_size,) + model.s_dim), \
                                               np.vstack(buffer_a), np.vstack(returns), np.vstack(adv)

        if render:
            self.env.render(statistics)

        return s_batch, a_batch, r_batch, adv_batch


def main():
    model_name = 'meta_ppo_model_shared_invest'
    ENVIRONMENT = 'PortfolioEnv'

    if ENVIRONMENT == 'PortfolioEnv':
        env = PortfolioEnv()
        main_env = PortfolioEnv()

    TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
    SUMMARY_DIR = os.path.join('./', 'PPO', ENVIRONMENT, TIMESTAMP)

    # BATCH = 128
    M = 256     # support set 길이
    K = 128     # target set 길이
    test_length = 20
    model = MetaPPO(env, M=M, K=K)

    # time.sleep(1)
    f_name = './{}_singletest.pkl'.format(model_name)
    if os.path.exists(f_name):
        model.load_model(f_name)

    n_envs = 1
    initial_t = 20      # 최초 랜덤선택을 위한 길이
    t_start = initial_t + M + K

    T = env.len_timeseries
    t = t_start
    t_step = 0
    s = main_env.reset(t)
    for t_step, t in enumerate(range(t_start, T - test_length, test_length)):
        print("t: {}".format(t))
        # env_samples = np.random.choice(M + np.arange(initial_t + t_step * test_length), n_envs, replace=True)
        env_samples = [268]

        # # train time
        EP_MAX = 50
        for ep in range(EP_MAX + 1):
            s_train = time.time()
            print("[TRAIN] t: {} / ep: {}".format(t, ep))
            model.metatrain(env_samples)
            e_train = time.time()
            print("[TRAIN] t: {} / ep: {} ({} sec per ep)".format(t, ep, e_train - s_train))

            if ep % 5 == 0:
                print('model saved. ({})'.format(f_name))
                model.save_model(f_name)

        # test time
        model.sampler.get_trajectories(model, env_samples[0], env_samples[0] + K, render=True, stochastic=False, statistics=True)

        test_buffer_r = []
        s = main_env.reset(env_samples[0])
        model.fast_train(env_samples[0])
        for n in range(test_length+100):
            s_test = time.time()
            print("[TEST] t: {} / n_day: {}".format(env_samples[0], n))
            a, v = model.evaluate_state(s, stochastic=False)
            s, r, _, done = main_env.step(np.squeeze(a))

            test_buffer_r.append(r)

            e_test = time.time()
            print("[TEST] t: {} / n_day: {} ({} sec)".format(t, n, e_test - s_test))

        main_env.render(statistics=True)
        data = main_env.sim.export()
