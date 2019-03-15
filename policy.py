import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers


def normalize(inputs, activation, reuse, scope):
    if args.norm == 'batch_norm':
        return tf_layers.batch_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'layer_norm':
        return tf_layers.layer_norm(inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif args.norm == 'None':
        if activation is not None:
            return activation(inputs)
        else:
            return inputs


class MyPolicy:
    def __init__(self, env, dim_hidden=[64]):
        self.dim_input = env.observation_space.shape
        self.dim_output = env.action_space.shape

        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden) + 1

    def construct_weights(self, scope_nm=None):
        if scope_nm is not None:
            with tf.variable_scope(scope_nm):
                weights = {}
                weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
                weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
                for i in range(1, len(self.dim_hidden)):
                    weights['w' + str(i + 1)] = tf.Variable(
                        tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
                    weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
                weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
                    tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
                weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward(self, input_data, weights, reuse=False):
        input_data = tf.reshape(input_data, [-1, self.dim_input])
        hidden = normalize(tf.matmul(input_data, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        out = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]
        return out

    def get_action(self, observation):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = deque()

    def add(self, s, a, r, s_, done):
        experience = (s, a, r, s_, done)
        if self.memory_counter < self.memory_size:
            self.memory.append(experience)
            self.memory_counter += 1
        else:
            self.memory.popleft()
            self.memory.append(experience)

    def clear(self):
        self.memory.clear()
        self.memory_counter = 0

    def sample_batch(self, batch_size):
        if self.memory_counter < batch_size:
            print('insufficient memory')
            return random.sample(self.memory, self.memory_counter)
            # return False
        else:
            return random.sample(self.memory, batch_size)


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
