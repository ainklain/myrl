
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
        return self.action_to_weight_linear(action), value_

    def action_to_weight_linear(self, action):
        a_positive = action - tf.math.reduce_min(action) + self.a_min
        return a_positive / tf.reduce_sum(a_positive, axis=1, keepdims=True)

    def action_to_weight_zero(self, action):
        a_positive = tf.clip_by_value(action, self.a_min, self.a_max)
        return a_positive

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

            print('{} / {} == loss: {}'.format(i, batch_count, batch_loss))

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

    def get_trajectories(self, model, begin_t=None, end_t=None, render=False):
        buffer_s, buffer_a, buffer_v, buffer_r, buffer_done = [], [], [], [], []
        rolling_r = RunningStats()

        s = self.env.reset(begin_t)
        done = False
        t_step = 0

        batch_size = end_t - begin_t
        while len(buffer_r) < batch_size:
            a, v = model.evaluate_state(s, stochastic=True)

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(tf.squeeze(v))
            buffer_done.append(done)

            a = tf.clip_by_value(a, self.env.action_space.low, self.env.action_space.high)
            s, r, _, done = self.env.step(np.squeeze(a))
            buffer_r.append(r)
            t_step += 1

            # print(t_step)
            time.sleep(1)
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
            self.env.render()

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

    time.sleep(1)
    f_name = './{}.pkl'.format(model_name)
    if os.path.exists(f_name):
        model.load_model(f_name)

    n_envs = 16
    initial_t = 20      # 최초 랜덤선택을 위한 길이
    t_start = initial_t + M + K

    T = env.len_timeseries
    t = t_start
    t_step = 0

    a_bt = []
    r_bt = []

    s = main_env.reset(t)
    for t_step, t in enumerate(range(t_start, T - test_length, test_length)):
        print("t: {}".format(t))
        env_samples = np.random.choice(M + np.arange(initial_t + t_step * test_length), n_envs, replace=True)

        # # train time
        # EP_MAX = 10
        # for ep in range(EP_MAX + 1):
        #     print("[TRAIN] t: {} / ep: {}".format(t, ep))
        #     model.metatrain(env_samples)
        #
        #     if ep % 10 == 0:
        #         model.save_model(f_name)

        # test time
        test_buffer_r = []
        for n in range(test_length):
            print("[TEST] t: {} / n_day: {}".format(t, n))
            model.fast_train(t)
            a, v = model.evaluate_state(s, stochastic=False)

            a = tf.clip_by_value(a, main_env.action_space.low, main_env.action_space.high)
            s, r, _, done = main_env.step(np.squeeze(a))

            test_buffer_r.append(r)
            if done:
                print('done')
                break

        if done:
            break











#
# class DDPG:
#     def __init__(self, sampler, args, action_noise=None):
#         self.gamma = args.gamma
#         self.sampler = sampler
#         self.args = args
#         self.action_noise = action_noise
#
#         self.actor_net = ActorNetwork(env=sampler.env, dim_hidden=args.dim_hidden_a)
#         self.critic_net = CriticNetwork(env=sampler.env, dim_hidden=args.dim_hidden_c)
#
#         self.target_actor_net = ActorNetwork(env=sampler.env, dim_hidden=args.dim_hidden_a)
#         self.target_critic_net = CriticNetwork(env=sampler.env, dim_hidden=args.dim_hidden_c)
#
#         self.target_actor_net.set_weights(self.actor_net.get_weights())
#         self.target_critic_net.set_weights(self.critic_net.get_weights())
#
#         self.inner_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#
#         self.optim_weight_a = None
#         self.optim_weight_c = None
#
#     def save_optim_weight(self):
#         self.optim_weight_a = self.actor_net.get_weights()
#         self.optim_weight_c = self.critic_net.get_weights()
#
#     def set_optim_weight_to_network(self, with_target):
#         self.actor_net.set_weights(self.optim_weight_a)
#         self.critic_net.set_weights(self.optim_weight_c)
#
#         if with_target:
#             self.target_actor_net.set_weights(self.optim_weight_a)
#             self.target_critic_net.set_weights(self.optim_weight_c)
#
#     def _trajectory_to_batch(self, trajectory, shuffle=True):
#         o_batch = np.zeros([len(trajectory)] + self.actor_net.dim_input, dtype=np.float32)
#         a_batch = np.zeros([len(trajectory), self.actor_net.dim_output], dtype=np.float32)
#         r_batch = np.zeros([len(trajectory), 1], dtype=np.float32)
#         o_batch_ = np.zeros_like(o_batch, dtype=np.float32)
#
#         if shuffle:
#             import random
#             random.shuffle(trajectory)
#
#         for i, transition in enumerate(trajectory):
#             o_batch[i] = self.process_obs(transition['obs'])
#             a_batch[i] = transition['action']
#             r_batch[i] = transition['reward']
#             o_batch_[i] = self.process_obs(transition['obs_'])
#
#         return o_batch, a_batch, r_batch, o_batch_
#
#     def set_action_noise(self, action_noise):
#         self.action_noise = action_noise
#
#     def get_trajectory(self, env_t, type_='support', training=True):
#         if type_ == 'support':
#             t_length = self.args.M
#         elif type_ == 'target':
#             t_length = self.args.K
#         else:
#             t_length = 1
#
#         tr, sim_data = self.sampler.sample_trajectory(self, env_t, t_length, action_noise=self.action_noise, training=training)
#
#         return tr, sim_data
#
#     def grad_loss(self, trajectory):
#         loss_c_object = tf.keras.losses.MeanSquaredError()
#
#         # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#
#         o_batch, a_batch, r_batch, o_batch_ = self._trajectory_to_batch(trajectory, shuffle=True)
#         o_processed = self.actor_net.shared_net(o_batch).numpy()
#         o_processed_ = self.target_actor_net.shared_net(o_batch_).numpy()
#         with tf.GradientTape() as tape_a:
#             with tf.GradientTape() as tape_c:
#                 q_target = r_batch + self.gamma * self.target_critic_net(o_processed_, self.target_actor_net(o_batch_))
#                 q_pred_for_c = self.critic_net(o_processed, a_batch)
#                 q_pred = self.critic_net(o_processed, self.actor_net(o_batch))
#
#                 loss_a = -tf.reduce_mean(q_pred)
#                 loss_c = loss_c_object(q_target, q_pred_for_c)
#
#         grad_loss_a = tape_a.gradient(loss_a, self.actor_net.trainable_variables)
#         grad_loss_c = tape_c.gradient(loss_c, self.critic_net.trainable_variables)
#
#         return grad_loss_a, grad_loss_c
#
#     def fast_train(self, tr_support):
#         # tr_support, _ = self.get_trajectory(env_t - self.args.M, 'support')
#         grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)
#
#         # print(self.actor_net.trainable_variables[0])
#         # print(grad_loss_support_a[0])
#         self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
#         self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))
#         # print(self.actor_net.trainable_variables[0])
#
#     def process_obs(self, observation):
#         if isinstance(observation, pd.DataFrame):
#             observation = observation.values
#         if len(observation.shape) == 2:
#             observation = np.expand_dims(observation, 0)
#         return observation
#
#     def get_action(self, observation):
#         observation = self.process_obs(observation)
#         return self.actor_net(observation).numpy()
#
#     def get_actions(self, observations):
#         if isinstance(observations, pd.DataFrame):
#             observations = observations.values
#
#         assert len(observations.shape) == 3, "value of first dim should be batch size"
#         return self.actor_net(observations).numpy()
#
#
#
# class MetaDDPG(DDPG):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initialize = False
#
#     def metatrain(self, env_samples):
#         meta_loss_a = None
#         meta_loss_c = None
#
#         # with tf.GradientTape() as tape_meta_a:
#         for env_t in env_samples:
#
#             self.set_optim_weight_to_network(with_target=True)
#
#             tr_support, _ = self.get_trajectory(env_t - self.args.M, 'support')
#             if not self.initialize:     # trajectory 첫 생성시 weight 초기화 되므로.
#                 self.optim_weight_a = self.actor_net.get_weights()
#                 self.optim_weight_c = self.critic_net.get_weights()
#                 self.initialize = True
#
#             grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)
#
#             self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
#             self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))
#
#             new_weight_a = self.actor_net.get_weights()
#             new_weight_c = self.critic_net.get_weights()
#
#             tr_target, _ = self.get_trajectory(env_t, 'target')
#
#             self.set_optim_weight_to_network(with_target=False)
#             grad_loss_a, grad_loss_c = self.grad_loss(tr_target)
#             if meta_loss_a is None:
#                 meta_loss_a = [grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
#                 meta_loss_c = [grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
#             else:
#                 meta_loss_a = [meta_loss_a[i] + grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
#                 meta_loss_c = [meta_loss_c[i] + grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
#
#         self.optimizer.apply_gradients(zip(meta_loss_a, self.actor_net.trainable_variables))
#         self.optimizer.apply_gradients(zip(meta_loss_c, self.critic_net.trainable_variables))
#
#
# class MetaDDPG2(DDPG):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def soft_update(self, tau=0.001):
#         theta_a = self.actor_net.get_weights()
#         theta_a_ = self.target_actor_net.get_weights()
#
#         theta_c = self.critic_net.get_weights()
#         theta_c_ = self.target_critic_net.get_weights()
#
#         new_target_weight_a = [theta_a[i] * tau + theta_a_[i] * (1-tau) for i in range(len(theta_a))]
#         new_target_weight_c = [theta_c[i] * tau + theta_c_[i] * (1 - tau) for i in range(len(theta_c))]
#
#         self.target_actor_net.set_weights(new_target_weight_a)
#         self.target_critic_net.set_weights(new_target_weight_c)
#
#     def metatrain(self, env_samples):
#         meta_loss_a = None
#         meta_loss_c = None
#
#         if self.optim_weight_a is None:
#             self.save_optim_weight()
#             self.set_optim_weight_to_network(with_target=True)
#
#         # with tf.GradientTape() as tape_meta_a:
#
#         cur_weight_a = self.actor_net.get_weights()
#         cur_weight_c = self.critic_net.get_weights()
#
#         for env_sample in env_samples:
#             env_t, tr_support, _ = env_sample
#
#             self.actor_net.set_weights(cur_weight_a)
#             self.critic_net.set_weights(cur_weight_c)
#
#             grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)
#
#             self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
#             self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))
#
#             # new_weight_a = self.actor_net.get_weights()
#             # new_weight_c = self.critic_net.get_weights()
#
#             tr_target, _ = self.get_trajectory(env_t, 'target')
#
#             self.actor_net.set_weights(cur_weight_a)
#             self.critic_net.set_weights(cur_weight_c)
#
#             grad_loss_a, grad_loss_c = self.grad_loss(tr_target)
#
#             if meta_loss_a is None:
#                 meta_loss_a = [grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
#                 meta_loss_c = [grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
#             else:
#                 meta_loss_a = [meta_loss_a[i] + grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
#                 meta_loss_c = [meta_loss_c[i] + grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
#
#         self.optimizer.apply_gradients(zip(meta_loss_a, self.actor_net.trainable_variables))
#         self.optimizer.apply_gradients(zip(meta_loss_c, self.critic_net.trainable_variables))
#
#         self.soft_update()
#
#         #
#         #
#         # with tf.GradientTape() as tape_a:
#         #     with tf.GradientTape() as tape_c:
#         #         # tape.watch([self.critic_net.trainable_variables, self.actor_net.trainable_variables])
#         #         for j, transition in enumerate(trajectory):
#         #             o = self.process_obs(transition['obs'])
#         #             # a = transition['action']
#         #             r = transition['reward']
#         #             o_ = self.process_obs(transition['obs_'])
#         #             q_target = r + self.gamma * self.target_critic_net(o_, self.target_actor_net)
#         #             loss_c_object = tf.keras.losses.MeanSquaredError()
#         #
#         #             # with tf.GradientTape() as tape:
#         #             q_pred = self.critic_net(o, self.actor_net)
#         #             if j == 0:
#         #                 loss_c = loss_c_object(q_target, q_pred) / len(trajectory)
#         #                 loss_a = -tf.reduce_mean(q_pred) / len(trajectory)
#         #             else:
#         #                 loss_c = loss_c + loss_c_object(q_target, q_pred) / len(trajectory)
#         #                 loss_a = loss_a - tf.reduce_mean(q_pred) / len(trajectory)
#         #
#         # grad_loss_c = tape_c.gradient(loss_c, self.critic_net.trainable_variables)
#         # grad_loss_a = tape_a.gradient(loss_a, self.actor_net.trainable_variables)
#
#         # random.shuffle(trajectory)
#         # total_loss_c = None
#         # for transition in trajectory:
#         #     o = self.process_obs(transition['obs'])
#         #     # a = transition['action']
#         #     r = transition['reward']
#         #     o_ = self.process_obs(transition['obs_'])
#         #     q_target = r + self.gamma * self.target_critic_net(o_, self.target_actor_net)
#         #     loss_c_object = tf.keras.losses.MeanSquaredError()
#         #
#         #     with tf.GradientTape() as tape:
#         #         q_pred = self.critic_net(o, self.actor_net)
#         #         loss_c = loss_c_object(q_target, q_pred)
#         #     grad_loss_c = tape.gradient(loss_c, self.critic_net.trainable_variables)
#         #
#         #     with tf.GradientTape() as tape:
#         #         q_pred = self.critic_net(o, self.actor_net)
#         #         loss_a = -tf.reduce_mean(q_pred)
#         #     grad_loss_a = tape.gradient(loss_a, self.actor_net.trainable_variables)
#         #
#         #     train_loss_a(grad_loss_a)
#         #     train_loss_c(grad_loss_c)
# # after:
# # t:80-0 Total Rewards: -0.003614 (instant: -0.000780 / delayed: -0.002833)   Total Cost: 62.03174 bp
# # t:80-0 strategy: nav: 0.997167, mean: -0.000008(0.000114) / std: 0.008904(0.008993)
# #
# # before:
# # t:80-0 Total Rewards: -0.007200 (instant: -0.003027 / delayed: -0.004173)   Total Cost: 66.28468 bp
# # t:80-0 strategy: nav: 0.997670, mean: -0.000011(0.000114) / std: 0.010869(0.008993)
#
#
#
# class MyPolicy:
#     def __init__(self, env, dim_hidden=[64, 32, 16]):
#         self.action_net = ActorNetwork(env, dim_hidden)
#
#     def get_action(self, observation):
#         if isinstance(observation, pd.DataFrame):
#             observation = observation.values
#         if len(observation.shape) == 2:
#             observation = np.expand_dims(observation, 0)
#         action = self.action_net(observation).numpy()
#         return self.value_to_weight(action)
#
#     def get_actions(self, observations):
#         if isinstance(observations, pd.DataFrame):
#             observations = observations.values
#
#         assert len(observations.shape) == 3, "value of first dim should be batch size"
#         actions = self.action_net(observations).numpy()
#         return self.value_to_weight(actions)
#
#     def value_to_weight(self, actions):
#         return actions / np.sum(actions, axis=1)
#
#
#
#
#
#
# # class Example(Model):
# #     def __init__(self):
# #         super().__init__()
# #         self.d1 = Dense(3, activation='relu')
# #
# #     def call(self, x):
# #         return self.d1(x)
# #
# # x = tf.cast(np.array([[1, 2, 3, 4]]), tf.float32)
# # model = Example()
# # y = model(x)
#
# # class MyPolicy(Model):
# #     def __init__(self, env, dim_hidden=[64]):
# #         super(MyPolicy, self).__init__()
# #         self.conv1 = Conv2D()
# #         self.dim_input = env.observation_space.shape
# #         self.dim_output = env.action_space.shape
# #
# #         self.dim_hidden = dim_hidden
# #         self.num_layers = len(dim_hidden) + 1
# #
# #         self.o_shared = self._build_shared_net(obs, 'o_shared', True)
# #         self.network_a = self._build_a(self.o_shared, 'network_a', True)
# #
# #     def construct_weights(self, scope):
# #         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# #             weights = {}
# #             weights['w1'] = tf.get_variable('w1', shape=[self.dim_input, self.dim_hidden[0]], initializer=tf.initializers.glorot_normal)
# #             weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
# #             for i in range(1, len(self.dim_hidden)):
# #                 weights['w' + str(i + 1)] = tf.Variable(
# #                     tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
# #                 weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
# #             weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
# #                 tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
# #             weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
# #         return weights
# #
# #     def _build_shared_net(self, obs, scope, trainable):
# #         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# #             obs_reshaped = tf.reshape(obs, [-1] + self.s_dim + [1])
# #             x = tf.layers.conv2d(obs_reshaped, filters=10, kernel_size=(20, 1), trainable=trainable)
# #             x = tf.layers.batch_normalization(x)
# #             x = tf.nn.relu(x)
# #             x = tf.layers.conv2d(x, filters=10, kernel_size=(1, 1), trainable=trainable)
# #             x = tf.layers.batch_normalization(x)
# #             x = tf.nn.relu(x)
# #             flattened_obs = tf.layers.flatten(x)
# #
# #             return flattened_obs
# #
# #     def _build_a(self, o_shared, scope, trainable):
# #         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# #             x = tf.layers.dense(o_shared, 64, activation=tf.nn.relu, trainable=trainable)
# #             x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=trainable)
# #             a = tf.layers.dense(x, self.a_dim, activation=tf.nn.softmax,
# #                                 kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003, seed=None),
# #                                 trainable=trainable)
# #
# #             return tf.multiply(a, self.a_bound, name='scaled_a')
# #
# #     def get_action(self, observation):
# #         sess = tf.get_default_graph()
# #         action = sess.run(self.network_a, feed_dict={observation})
# #         return action
# #
# #     def get_actions(self, observations):
# #         raise NotImplementedError
#
#
#
#
# class DDPG(object):
#     def __init__(self, sess, a_dim, s_dim, a_bound, memory, gamma, tau, lr_a, lr_c, batch_size, ):
#         self.sess = sess
#         self.a_dim = a_dim
#         self.a_bound = a_bound
#         self.s_dim = s_dim
#         self.memory = memory
#         self.batch_size = batch_size
#
#         self.S = tf.placeholder(tf.float32, [None] + s_dim, 's')
#         self.S_ = tf.placeholder(tf.float32, [None] + s_dim, 's_')
#         self.R = tf.placeholder(tf.float32, [None, 1], 'r')
#
#         with tf.variable_scope('Shared'):
#             o_shared = self._build_shared_net(self.S, scope='eval', trainable=True)
#             o_shared_ = self._build_shared_net(self.S_, scope='target', trainable=False)
#         with tf.variable_scope('Actor'):
#             self.a = self._build_a(o_shared, scope='eval', trainable=True)
#             a_ = self._build_a(o_shared_, scope='target', trainable=False)
#         with tf.variable_scope('Critic'):
#             q = self._build_c(o_shared, self.a, scope='eval', trainable=True)
#             q_ = self._build_c(o_shared_, a_, scope='target', trainable=False)
#
#         self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/eval')
#         self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/target')
#         self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
#         self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
#         self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
#         self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
#
#         self.soft_replace = [tf.assign(t, (1-tau) * t + tau * e)
#                              for t, e in zip(self.at_params + self.ct_params + self.st_params,
#                                              self.ae_params + self.ce_params + self.se_params)]
#
#         q_target = self.R + gamma * q_
#
#         td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
#         self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)
#
#         a_loss = -tf.reduce_mean(q)
#         self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)
#
#         self.sess.run(tf.global_variables_initializer())
#
#     def choose_action(self, s):
#         return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
#
#     def learn(self):
#         self.sess.run(self.soft_replace)
#
#         mini_batch = self.memory.sample_batch(self.batch_size)
#         if not mini_batch:
#             return False
#
#         s, a, r, s_, done = [np.array([i_row[j] for i_row in mini_batch])
#                              for j in range(5)]
#
#         self.sess.run(self.atrain, {self.S: s})
#         self.sess.run(self.ctrain, {self.S: s, self.a: a, self.R: r, self.S_: s_})
#
#     def store_transition(self, s, a, r, s_, done):
#         self.memory.add(s, a, r, s_, done)
#
#     def _build_shared_net(self, s, scope, trainable):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             s_reshaped = tf.reshape(s, [-1] + self.s_dim + [1])
#             x = tf.layers.conv2d(s_reshaped, filters=10, kernel_size=(30, 1), trainable=trainable)
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
#             x = tf.layers.conv2d(x, filters=10, kernel_size=(1, 1), trainable=trainable)
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
#             flattened_obs = tf.layers.flatten(x)
#
#             return flattened_obs
#
#     def _build_a(self, o_shared, scope, trainable):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             x = tf.layers.dense(o_shared, 64, activation=tf.nn.relu, trainable=trainable)
#             x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=trainable)
#             a = tf.layers.dense(x, self.a_dim, activation=tf.nn.softmax,
#                                 kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003, seed=None),
#                                 trainable=trainable)
#
#             return tf.multiply(a, self.a_bound, name='scaled_a')
#
#     def _build_c(self, o_shared, a, scope, trainable):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             n_l1 = 30
#             w1_s = tf.get_variable('w1_s', [o_shared.shape.as_list()[1], n_l1], trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
#             b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
#             net = tf.nn.relu(tf.matmul(o_shared, w1_s) + tf.matmul(a, w1_a) + b1)
#             net = tf.layers.dense(net, 1, trainable=trainable)
#             return net
