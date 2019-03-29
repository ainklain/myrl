
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, BatchNormalization, ReLU


class SharedNetwork(Model):
    def __init__(self, env):
        super(SharedNetwork, self).__init__()
        self.dim_input = list(env.observation_space.shape)
        self.dim_output = env.action_space.shape[0]
        self.reshape = Reshape(self.dim_input + [1])

        self.relu = ReLU()
        self.flatten = Flatten()
        self.batch_norm = BatchNormalization()

    def build_convnet(self, x, len_filter=20):
        conv1 = Conv2D(8, (len_filter, 1), dtype=tf.float32, padding='same')
        conv2 = Conv2D(4, (self.dim_input[0], 1), dtype=tf.float32)
        x = conv1(x)
        x = self.batch_norm(x)
        # x = self.relu(x)
        x = conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.flatten(x)

        return x

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.reshape(x)

        x_20 = self.build_convnet(x, 20)
        x_60 = self.build_convnet(x, 60)

        x = tf.concat([x_20, x_60], axis=1)

        return x


class ActorNetwork(Model):
    def __init__(self, env, dim_hidden=[64, 32, 16]):
        super(ActorNetwork, self).__init__()
        self.shared_net = SharedNetwork(env)
        self.dim_hidden = dim_hidden

        self.hidden_layer = dict()
        for i, dim_h in enumerate(dim_hidden):
            self.hidden_layer['h' + str(i + 1)] = Dense(dim_h, activation='relu')

        dim_output = self.shared_net.dim_output
        self.output_layer = Dense(dim_output, activation='sigmoid')

    @property
    def dim_input(self):
        return self.shared_net.dim_input

    @property
    def dim_output(self):
        return self.shared_net.dim_output

    def call(self, x):
        x = self.shared_net(x)
        for i in range(len(self.dim_hidden)):
            x = self.hidden_layer['h' + str(i + 1)](x)
        x = self.output_layer(x)
        return x / tf.reduce_sum(x, axis=1, keepdims=True)


class CriticNetwork(Model):
    def __init__(self, env, dim_hidden=[32]):
        super(CriticNetwork, self).__init__(env)
        self.dim_hidden = dim_hidden

        self.hidden_layer = dict()
        for i, dim_h in enumerate(dim_hidden):
            self.hidden_layer['hs' + str(i + 1)] = Dense(dim_h, activation='linear', use_bias=False)
            self.hidden_layer['ha' + str(i + 1)] = Dense(dim_h, activation='linear')
        self.relu = ReLU()
        self.output_layer = Dense(1, activation='linear')

    def call(self, x_processed, action):
        for i in range(len(self.dim_hidden)):
            x_processed = self.hidden_layer['hs' + str(i + 1)](x_processed)
            action = self.hidden_layer['ha' + str(i + 1)](action)
            x = self.relu(tf.math.add(x_processed, action))

        return self.output_layer(x)


class DDPG:
    def __init__(self, sampler, args, action_noise=None):
        self.gamma = args.gamma
        self.sampler = sampler
        self.args = args
        self.action_noise = action_noise

        self.actor_net = ActorNetwork(env=sampler.env, dim_hidden=args.dim_hidden_a)
        self.critic_net = CriticNetwork(env=sampler.env, dim_hidden=args.dim_hidden_c)

        self.target_actor_net = ActorNetwork(env=sampler.env, dim_hidden=args.dim_hidden_a)
        self.target_critic_net = CriticNetwork(env=sampler.env, dim_hidden=args.dim_hidden_c)

        self.target_actor_net.set_weights(self.actor_net.get_weights())
        self.target_critic_net.set_weights(self.critic_net.get_weights())

        self.inner_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

        self.optim_weight_a = None
        self.optim_weight_c = None

    def save_optim_weight(self):
        self.optim_weight_a = self.actor_net.get_weights()
        self.optim_weight_c = self.critic_net.get_weights()

    def set_optim_weight_to_network(self, with_target):
        self.actor_net.set_weights(self.optim_weight_a)
        self.critic_net.set_weights(self.optim_weight_c)

        if with_target:
            self.target_actor_net.set_weights(self.optim_weight_a)
            self.target_critic_net.set_weights(self.optim_weight_c)

    def _trajectory_to_batch(self, trajectory):
        o_batch = np.zeros([len(trajectory)] + self.actor_net.dim_input, dtype=np.float32)
        a_batch = np.zeros([len(trajectory), self.actor_net.dim_output], dtype=np.float32)
        r_batch = np.zeros([len(trajectory), 1], dtype=np.float32)
        o_batch_ = np.zeros_like(o_batch, dtype=np.float32)

        for i, transition in enumerate(trajectory):
            o_batch[i] = self.process_obs(transition['obs'])
            a_batch[i] = transition['action']
            r_batch[i] = transition['reward']
            o_batch_[i] = self.process_obs(transition['obs_'])

        return o_batch, a_batch, r_batch, o_batch_

    def set_action_noise(self, action_noise):
        self.action_noise = action_noise

    def get_trajectory(self, env_t, type_='support', training=True):
        if type_ == 'support':
            t_length = self.args.M
        elif type_ == 'target':
            t_length = self.args.K
        else:
            t_length = 1

        tr, sim_data = self.sampler.sample_trajectory(self, env_t, t_length, action_noise=self.action_noise, training=training)

        return tr, sim_data

    def grad_loss(self, trajectory):
        loss_c_object = tf.keras.losses.MeanSquaredError()

        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        o_batch, a_batch, r_batch, o_batch_ = self._trajectory_to_batch(trajectory)
        o_processed = self.actor_net.shared_net(o_batch).numpy()
        o_processed_ = self.target_actor_net.shared_net(o_batch_).numpy()
        with tf.GradientTape() as tape_a:
            with tf.GradientTape() as tape_c:
                q_target = r_batch + self.gamma * self.target_critic_net(o_processed_, self.target_actor_net(o_batch_))
                q_pred_for_c = self.critic_net(o_processed, a_batch)
                q_pred = self.critic_net(o_processed, self.actor_net(o_batch))

                loss_a = -tf.reduce_mean(q_pred)
                loss_c = loss_c_object(q_target, q_pred_for_c)

        grad_loss_a = tape_a.gradient(loss_a, self.actor_net.trainable_variables)
        grad_loss_c = tape_c.gradient(loss_c, self.critic_net.trainable_variables)

        return grad_loss_a, grad_loss_c

    def fast_train(self, tr_support):
        # tr_support, _ = self.get_trajectory(env_t - self.args.M, 'support')
        grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)

        # print(self.actor_net.trainable_variables[0])
        # print(grad_loss_support_a[0])
        self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
        self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))
        # print(self.actor_net.trainable_variables[0])

    def process_obs(self, observation):
        if isinstance(observation, pd.DataFrame):
            observation = observation.values
        if len(observation.shape) == 2:
            observation = np.expand_dims(observation, 0)
        return observation

    def get_action(self, observation):
        observation = self.process_obs(observation)
        return self.actor_net(observation).numpy()

    def get_actions(self, observations):
        if isinstance(observations, pd.DataFrame):
            observations = observations.values

        assert len(observations.shape) == 3, "value of first dim should be batch size"
        return self.actor_net(observations).numpy()



class MetaDDPG(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize = False

    def metatrain(self, env_samples):
        meta_loss_a = None
        meta_loss_c = None

        # with tf.GradientTape() as tape_meta_a:
        for env_t in env_samples:

            self.set_optim_weight_to_network(with_target=True)

            tr_support, _ = self.get_trajectory(env_t - self.args.M, 'support')
            if not self.initialize:     # trajectory 첫 생성시 weight 초기화 되므로.
                self.optim_weight_a = self.actor_net.get_weights()
                self.optim_weight_c = self.critic_net.get_weights()
                self.initialize = True

            grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)

            self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
            self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))

            new_weight_a = self.actor_net.get_weights()
            new_weight_c = self.critic_net.get_weights()

            tr_target, _ = self.get_trajectory(env_t, 'target')

            self.set_optim_weight_to_network(with_target=False)
            grad_loss_a, grad_loss_c = self.grad_loss(tr_target)
            if meta_loss_a is None:
                meta_loss_a = [grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
                meta_loss_c = [grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
            else:
                meta_loss_a = [meta_loss_a[i] + grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
                meta_loss_c = [meta_loss_c[i] + grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]

        self.optimizer.apply_gradients(zip(meta_loss_a, self.actor_net.trainable_variables))
        self.optimizer.apply_gradients(zip(meta_loss_c, self.critic_net.trainable_variables))


class MetaDDPG2(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def soft_update(self, tau=0.001):
        theta_a = self.actor_net.get_weights()
        theta_a_ = self.target_actor_net.get_weights()

        theta_c = self.critic_net.get_weights()
        theta_c_ = self.target_critic_net.get_weights()

        new_target_weight_a = [theta_a[i] * tau + theta_a_[i] * (1-tau) for i in range(len(theta_a))]
        new_target_weight_c = [theta_c[i] * tau + theta_c_[i] * (1 - tau) for i in range(len(theta_c))]

        self.target_actor_net.set_weights(new_target_weight_a)
        self.target_critic_net.set_weights(new_target_weight_c)

    def metatrain(self, env_samples):
        meta_loss_a = None
        meta_loss_c = None

        if self.optim_weight_a is None:
            self.save_optim_weight()
            self.set_optim_weight_to_network(with_target=True)

        # with tf.GradientTape() as tape_meta_a:

        cur_weight_a = self.actor_net.get_weights()
        cur_weight_c = self.critic_net.get_weights()

        for env_sample in env_samples:
            env_t, tr_support, _ = env_sample

            self.actor_net.set_weights(cur_weight_a)
            self.critic_net.set_weights(cur_weight_c)

            grad_loss_support_a, grad_loss_support_c = self.grad_loss(tr_support)

            self.inner_optimizer.apply_gradients(zip(grad_loss_support_a, self.actor_net.trainable_variables))
            self.inner_optimizer.apply_gradients(zip(grad_loss_support_c, self.critic_net.trainable_variables))

            # new_weight_a = self.actor_net.get_weights()
            # new_weight_c = self.critic_net.get_weights()

            tr_target, _ = self.get_trajectory(env_t, 'target')

            self.actor_net.set_weights(cur_weight_a)
            self.critic_net.set_weights(cur_weight_c)

            grad_loss_a, grad_loss_c = self.grad_loss(tr_target)

            if meta_loss_a is None:
                meta_loss_a = [grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
                meta_loss_c = [grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]
            else:
                meta_loss_a = [meta_loss_a[i] + grad_loss_a[i] / len(env_samples) for i in range(len(grad_loss_a))]
                meta_loss_c = [meta_loss_c[i] + grad_loss_c[i] / len(env_samples) for i in range(len(grad_loss_c))]

        self.optimizer.apply_gradients(zip(meta_loss_a, self.actor_net.trainable_variables))
        self.optimizer.apply_gradients(zip(meta_loss_c, self.critic_net.trainable_variables))

        self.soft_update()

        #
        #
        # with tf.GradientTape() as tape_a:
        #     with tf.GradientTape() as tape_c:
        #         # tape.watch([self.critic_net.trainable_variables, self.actor_net.trainable_variables])
        #         for j, transition in enumerate(trajectory):
        #             o = self.process_obs(transition['obs'])
        #             # a = transition['action']
        #             r = transition['reward']
        #             o_ = self.process_obs(transition['obs_'])
        #             q_target = r + self.gamma * self.target_critic_net(o_, self.target_actor_net)
        #             loss_c_object = tf.keras.losses.MeanSquaredError()
        #
        #             # with tf.GradientTape() as tape:
        #             q_pred = self.critic_net(o, self.actor_net)
        #             if j == 0:
        #                 loss_c = loss_c_object(q_target, q_pred) / len(trajectory)
        #                 loss_a = -tf.reduce_mean(q_pred) / len(trajectory)
        #             else:
        #                 loss_c = loss_c + loss_c_object(q_target, q_pred) / len(trajectory)
        #                 loss_a = loss_a - tf.reduce_mean(q_pred) / len(trajectory)
        #
        # grad_loss_c = tape_c.gradient(loss_c, self.critic_net.trainable_variables)
        # grad_loss_a = tape_a.gradient(loss_a, self.actor_net.trainable_variables)

        # random.shuffle(trajectory)
        # total_loss_c = None
        # for transition in trajectory:
        #     o = self.process_obs(transition['obs'])
        #     # a = transition['action']
        #     r = transition['reward']
        #     o_ = self.process_obs(transition['obs_'])
        #     q_target = r + self.gamma * self.target_critic_net(o_, self.target_actor_net)
        #     loss_c_object = tf.keras.losses.MeanSquaredError()
        #
        #     with tf.GradientTape() as tape:
        #         q_pred = self.critic_net(o, self.actor_net)
        #         loss_c = loss_c_object(q_target, q_pred)
        #     grad_loss_c = tape.gradient(loss_c, self.critic_net.trainable_variables)
        #
        #     with tf.GradientTape() as tape:
        #         q_pred = self.critic_net(o, self.actor_net)
        #         loss_a = -tf.reduce_mean(q_pred)
        #     grad_loss_a = tape.gradient(loss_a, self.actor_net.trainable_variables)
        #
        #     train_loss_a(grad_loss_a)
        #     train_loss_c(grad_loss_c)
# after:
# t:80-0 Total Rewards: -0.003614 (instant: -0.000780 / delayed: -0.002833)   Total Cost: 62.03174 bp
# t:80-0 strategy: nav: 0.997167, mean: -0.000008(0.000114) / std: 0.008904(0.008993)
#
# before:
# t:80-0 Total Rewards: -0.007200 (instant: -0.003027 / delayed: -0.004173)   Total Cost: 66.28468 bp
# t:80-0 strategy: nav: 0.997670, mean: -0.000011(0.000114) / std: 0.010869(0.008993)



class MyPolicy:
    def __init__(self, env, dim_hidden=[64, 32, 16]):
        self.action_net = ActorNetwork(env, dim_hidden)

    def get_action(self, observation):
        if isinstance(observation, pd.DataFrame):
            observation = observation.values
        if len(observation.shape) == 2:
            observation = np.expand_dims(observation, 0)
        action = self.action_net(observation).numpy()
        return self.value_to_weight(action)

    def get_actions(self, observations):
        if isinstance(observations, pd.DataFrame):
            observations = observations.values

        assert len(observations.shape) == 3, "value of first dim should be batch size"
        actions = self.action_net(observations).numpy()
        return self.value_to_weight(actions)

    def value_to_weight(self, actions):
        return actions / np.sum(actions, axis=1)






# class Example(Model):
#     def __init__(self):
#         super().__init__()
#         self.d1 = Dense(3, activation='relu')
#
#     def call(self, x):
#         return self.d1(x)
#
# x = tf.cast(np.array([[1, 2, 3, 4]]), tf.float32)
# model = Example()
# y = model(x)

# class MyPolicy(Model):
#     def __init__(self, env, dim_hidden=[64]):
#         super(MyPolicy, self).__init__()
#         self.conv1 = Conv2D()
#         self.dim_input = env.observation_space.shape
#         self.dim_output = env.action_space.shape
#
#         self.dim_hidden = dim_hidden
#         self.num_layers = len(dim_hidden) + 1
#
#         self.o_shared = self._build_shared_net(obs, 'o_shared', True)
#         self.network_a = self._build_a(self.o_shared, 'network_a', True)
#
#     def construct_weights(self, scope):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             weights = {}
#             weights['w1'] = tf.get_variable('w1', shape=[self.dim_input, self.dim_hidden[0]], initializer=tf.initializers.glorot_normal)
#             weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
#             for i in range(1, len(self.dim_hidden)):
#                 weights['w' + str(i + 1)] = tf.Variable(
#                     tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
#                 weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
#             weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
#                 tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
#             weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
#         return weights
#
#     def _build_shared_net(self, obs, scope, trainable):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             obs_reshaped = tf.reshape(obs, [-1] + self.s_dim + [1])
#             x = tf.layers.conv2d(obs_reshaped, filters=10, kernel_size=(20, 1), trainable=trainable)
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
#     def get_action(self, observation):
#         sess = tf.get_default_graph()
#         action = sess.run(self.network_a, feed_dict={observation})
#         return action
#
#     def get_actions(self, observations):
#         raise NotImplementedError




class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound, memory, gamma, tau, lr_a, lr_c, batch_size, ):
        self.sess = sess
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.s_dim = s_dim
        self.memory = memory
        self.batch_size = batch_size

        self.S = tf.placeholder(tf.float32, [None] + s_dim, 's')
        self.S_ = tf.placeholder(tf.float32, [None] + s_dim, 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Shared'):
            o_shared = self._build_shared_net(self.S, scope='eval', trainable=True)
            o_shared_ = self._build_shared_net(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Actor'):
            self.a = self._build_a(o_shared, scope='eval', trainable=True)
            a_ = self._build_a(o_shared_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(o_shared, self.a, scope='eval', trainable=True)
            q_ = self._build_c(o_shared_, a_, scope='target', trainable=False)

        self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/eval')
        self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Shared/target')
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [tf.assign(t, (1-tau) * t + tau * e)
                             for t, e in zip(self.at_params + self.ct_params + self.st_params,
                                             self.ae_params + self.ce_params + self.se_params)]

        q_target = self.R + gamma * q_

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        mini_batch = self.memory.sample_batch(self.batch_size)
        if not mini_batch:
            return False

        s, a, r, s_, done = [np.array([i_row[j] for i_row in mini_batch])
                             for j in range(5)]

        self.sess.run(self.atrain, {self.S: s})
        self.sess.run(self.ctrain, {self.S: s, self.a: a, self.R: r, self.S_: s_})

    def store_transition(self, s, a, r, s_, done):
        self.memory.add(s, a, r, s_, done)

    def _build_shared_net(self, s, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            s_reshaped = tf.reshape(s, [-1] + self.s_dim + [1])
            x = tf.layers.conv2d(s_reshaped, filters=10, kernel_size=(30, 1), trainable=trainable)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=10, kernel_size=(1, 1), trainable=trainable)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            flattened_obs = tf.layers.flatten(x)

            return flattened_obs

    def _build_a(self, o_shared, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(o_shared, 64, activation=tf.nn.relu, trainable=trainable)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=trainable)
            a = tf.layers.dense(x, self.a_dim, activation=tf.nn.softmax,
                                kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003, seed=None),
                                trainable=trainable)

            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, o_shared, a, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [o_shared.shape.as_list()[1], n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(o_shared, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 1, trainable=trainable)
            return net
