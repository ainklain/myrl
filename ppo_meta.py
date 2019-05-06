# conda install swig # needed to build box2d in the pip install
# pip install box2d-py # a repackaged version of pybox2d


from test_rl.distribution import DiagonalGaussian

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from time import time
import gym
import scipy
import pickle

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

EP_MAX = 1000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01
LR = 0.0001
META_LR = 0.001
# BATCH = 8192
BATCH = 2048
MINIBATCH = 32
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
SIGMA_FLOOR = 0.0
model_name = 'model_shared'


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
        super().__init__()
        self.conv1 = Conv2D(32, kernel_size=8, strides=4, dtype=tf.float32, activation=tf.nn.relu)
        self.conv2 = Conv2D(64, kernel_size=4, strides=2, dtype=tf.float32, activation=tf.nn.relu)
        self.conv3 = Conv2D(64, kernel_size=3, strides=1, dtype=tf.float32, activation=tf.nn.relu)
        self.flatten = Flatten()

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, 0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x


class ParamNetwork(Model):
    def __init__(self, a_dim):
        super().__init__()
        self.feature_net = FeatureNetwork()
        self.dense1 = Dense(400, activation=tf.nn.relu, kernel_regularizer=l2(L2_REG))
        self.dense2 = Dense(400, activation=tf.nn.relu, kernel_regularizer=l2(L2_REG))
        self.dense_mu = Dense(a_dim, activation=tf.nn.tanh, kernel_regularizer=l2(L2_REG))
        self.dense_critic = Dense(1, kernel_regularizer=l2(L2_REG))

    def call(self, x):
        x = self.feature_net(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense_mu(x), self.dense_critic(x)


class PPO(object):
    def __init__(self, env):
        self.discrete = False
        self.s_dim = env.observation_space.shape
        if len(env.observation_space.shape) > 0:
            self.a_dim = env.action_space.shape[0]
            self.a_bound = (env.action_space.high - env.action_space.low) / 2
        else:
            self.a_dim = env.action_space.n

        self.dist = DiagonalGaussian(self.a_dim)

        self.param_network = ParamNetwork(self.a_dim)
        self.log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.old_param_network = ParamNetwork(self.a_dim)
        self.old_log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.optimizer = tf.optimizers.Adam(LR)

        self.global_step = 0

        self._initialize(env.reset())

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
        return action, value_

    def polynomial_epsilon_decay(self, learning_rate, global_step, decay_steps, end_learning_rate, power):
        global_step_ = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step_ / decay_steps) ** (power) \
                                + end_learning_rate

        return decayed_learning_rate

    def update(self, s_batch, a_batch, r_batch, adv_batch):
        start = time()
        e_time = []

        self.assign_old_network()

        for epoch in range(EPOCHS):
            idx = np.arange(len(s_batch))
            np.random.shuffle(idx)

            loss_per_epoch = 0
            for i in range(len(s_batch) // MINIBATCH):
                epsilon_decay = self.polynomial_epsilon_decay(0.1, self.global_step, 1e5, 0.01, power=1.0)
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

                    loss = loss_pi + loss_v * VF_COEFF + pol_entpen

                grad = tape.gradient(loss, self.param_network.trainable_variables + [self.log_sigma])
                self.optimizer.apply_gradients(zip(grad, self.param_network.trainable_variables + [self.log_sigma]))

                loss_per_epoch = loss_per_epoch + loss
                # print("epoch: {} - {}/{} ({:.3f}%),  loss: {:.8f}".format(epoch, i, len(s_batch) // MINIBATCH,
                #                                                           i / (len(s_batch) // MINIBATCH) * 100., loss))
                # if i % 10 == 0:
                #     print(grad[-1])

                self.global_step += 1

            print("epoch: {} - loss: {}".format(epoch, loss_per_epoch / (len(s_batch) // MINIBATCH) * 100))

    def save_model(self, f_name):
        w_dict = {}
        w_dict['param_network'] = self.param_network.get_weights()
        w_dict['log_sigma'] = self.log_sigma.numpy()
        w_dict['global_step'] = self.global_step

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)
        self.param_network.set_weights(w_dict['param_network'])
        self.log_sigma.assign(w_dict['log_sigma'])
        self.global_step = w_dict['global_step']

        print("model loaded. (path: {})".format(f_name))


class MetaPPO(object):
    def __init__(self, env, M=256, K=128):
        self.discrete = False
        self.s_dim = env.observation_space.shape
        if len(env.observation_space.shape) > 0:
            self.a_dim = env.action_space.shape[0]
            self.a_bound = (env.action_space.high - env.action_space.low) / 2
        else:
            self.a_dim = env.action_space.n

        self.dist = DiagonalGaussian(self.a_dim)

        self.param_network = ParamNetwork(self.a_dim)
        self.log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.old_param_network = ParamNetwork(self.a_dim)
        self.old_log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.meta_optimizer = tf.optimizers.Adam(META_LR)
        self.optimizer = tf.optimizers.Adam(LR)

        self.sampler = Sampler(env, M=M, K=K)

        self.M = M
        self.K = K

        self.global_step = 0
        self._initialize(env.reset())

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
        return action, value_

    def polynomial_epsilon_decay(self, learning_rate, global_step, decay_steps, end_learning_rate, power):
        global_step_ = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step_ / decay_steps) ** (power) \
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

            print('{} / {} == loss: {}'.format(i, batch_count, batch_loss))

        return grad

    def meta_update(self, n_envs):
        self.assign_old_network()
        meta_grad = None

        param_network_wgt = self.param_network.get_weights()
        log_sigma_wgt = self.log_sigma.numpy()

        for env_i in range(n_envs):
            self.param_network.set_weights(param_network_wgt)
            self.log_sigma.assign(log_sigma_wgt)

            trajectory_sample = self.sampler.get_trajectories(self, support=True)
            grad = self.grad_loss(trajectory_sample)
            self.optimizer.apply_gradients(zip(grad, self.param_network.trainable_variables + [self.log_sigma]))

            trajectory_target = self.sampler.get_trajectories(self, support=False)

            self.param_network.set_weights(param_network_wgt)
            self.log_sigma.assign(log_sigma_wgt)

            grad = self.grad_loss(trajectory_target)
            print("env_i: {} / {}".format(env_i, n_envs))
            if meta_grad is None:
                meta_grad = [grad[k] / n_envs for k in range(len(grad))]
            else:
                meta_grad = [meta_grad[k] + grad[k] / n_envs for k in range(len(grad))]

        self.meta_optimizer.apply_gradients(zip(meta_grad, self.param_network.trainable_variables + [self.log_sigma]))

    def update(self, s_batch, a_batch, r_batch, adv_batch):
        start = time()
        e_time = []

        self.assign_old_network()

        for epoch in range(EPOCHS):
            idx = np.arange(len(s_batch))
            np.random.shuffle(idx)

            loss_per_epoch = 0
            for i in range(len(s_batch) // MINIBATCH):
                epsilon_decay = self.polynomial_epsilon_decay(0.1, self.global_step, 1e5, 0.01, power=1.0)
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

                    loss = loss_pi + loss_v * VF_COEFF + pol_entpen

                grad = tape.gradient(loss, self.param_network.trainable_variables + [self.log_sigma])
                self.optimizer.apply_gradients(zip(grad, self.param_network.trainable_variables + [self.log_sigma]))

                loss_per_epoch = loss_per_epoch + loss
                # print("epoch: {} - {}/{} ({:.3f}%),  loss: {:.8f}".format(epoch, i, len(s_batch) // MINIBATCH,
                #                                                           i / (len(s_batch) // MINIBATCH) * 100., loss))
                # if i % 10 == 0:
                #     print(grad[-1])

                self.global_step += 1

            print("epoch: {} - loss: {}".format(epoch, loss_per_epoch / (len(s_batch) // MINIBATCH) * 100))

    def save_model(self, f_name):
        w_dict = {}
        w_dict['param_network'] = self.param_network.get_weights()
        w_dict['log_sigma'] = self.log_sigma.numpy()
        w_dict['global_step'] = self.global_step

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)
        self.param_network.set_weights(w_dict['param_network'])
        self.log_sigma.assign(w_dict['log_sigma'])
        self.global_step = w_dict['global_step']

        print("model loaded. (path: {})".format(f_name))


class Sampler:
    def __init__(self, env, M, K):
        self.env = env
        self.M = M
        self.K = K

    def get_trajectories(self, model, support=True):
        buffer_s, buffer_a, buffer_v, buffer_r, buffer_done = [], [], [], [], []
        rolling_r = RunningStats()

        if support is True:
            s = self.env.reset()
            s = s / 255.
            batch_size = self.M
        else:
            s = self.last_s
            batch_size = self.K

        self.env.render()
        done = False

        while len(buffer_r) < batch_size:
            a, v = model.evaluate_state(s, stochastic=True)
            self.env.render()

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(tf.squeeze(v))
            buffer_done.append(done)

            a = tf.clip_by_value(a, self.env.action_space.low, self.env.action_space.high)
            s, r, done, _ = self.env.step(np.squeeze(a))
            s = s / 255.
            buffer_r.append(r)

            self.last_s = s

            if done:
                s = self.env.reset()
                done = False

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

        return s_batch, a_batch, r_batch, adv_batch


def main():
    # ENVIRONMENT = 'Pendulum-v0'
    ENVIRONMENT = 'CarRacing-v0'
    # ENVIRONMENT = 'Acrobot-v1'

    # from gym.envs.box2d.car_racing import CarRacing
    # env = CarRacing()
    env = gym.make(ENVIRONMENT)

    TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
    SUMMARY_DIR = os.path.join('./', 'PPO', ENVIRONMENT, TIMESTAMP)

    M = 512
    K = 512
    model = MetaPPO(env, M=M, K=K)
    model_ppo = PPO(env)

    f_name = './{}_meta2.pkl'.format(model_name)
    if os.path.exists(f_name):
        model.load_model(f_name)

    f_name_ppo = './{}.pkl'.format(model_name)
    if os.path.exists(f_name_ppo):
        model_ppo.load_model(f_name_ppo)

    n_envs = 4
    t, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []

    rolling_r = RunningStats()

    EP_MAX = 1000
    for episode in range(EP_MAX + 1):
        print(episode)
        model.meta_update(n_envs)
        if episode % 10 == 0:
            model.save_model(f_name)
    env.close()



    # env = gym.make(ENVIRONMENT)
    # ppo = PPO(env)
    # if os.path.exists(f_name):
    #     ppo.load_model(f_name)
    #
    # t, terminal = 0, False
    # for episode in range(5 + 1):
    #     print(episode)
    #     s = env.reset()
    #     s = s / 255.
    #     env.render()
    #     ep_r, ep_t, ep_a = 0, 0, []
    #
    #     while True:
    #         a, v = ppo.evaluate_state(s, stochastic=False)
    #         env.render()
    #
    #         if not ppo.discrete:
    #             a = tf.clip_by_value(a, env.action_space.low, env.action_space.high)
    #         s, r, terminal, _ = env.step(np.squeeze(a))
    #         s = s / 255.
    #         buffer_r.append(r)
    #         ep_r += r
    #         ep_t += 1
    #         t += 1
    #
    #         if terminal:
    #             break
