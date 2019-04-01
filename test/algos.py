from test.base import RLAlgorithm

import numpy as np
import tensorflow as tf
import time

class BatchMAMLPolopt(RLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 baseline,
                 scope=None,
                 n_itr=500,
                 start_itr=0,
                 batch_size=100,
                 max_path_length=500,
                 meta_batch_size=100,
                 num_grad_updates=1,
                 discount=0.99,
                 gae_lambda=1,
                 plot=False,
                 pause_for_plot=False,
                 center_adv=True,
                 positive_adv=False,
                 store_paths=True,
                 whole_paths=True,
                 fixed_horizon=False,
                 sampler_cls=None,
                 sampler_args=None,
                 force_batch_sampler=False,
                 use_maml=True,
                 load_policy=None,
                 **kwargs):
        self.env = env
        self.policy = policy
        self.load_policy = load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount= discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size
        self.num_grad_updates = num_grad_updates

        if sampler_cls is None:
            sampler_cls = BatchSampler

        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

    def init_opt(self):
        raise NotImplementedError

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', log=True):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log)

    def log_diagnostics(self, paths, prefix):
        pass
        # self.env.log_diagnostics(paths, prefix)
        # self.policy.log_diagnostics(paths, prefix)
        # self.baseline.log_diagnostics(paths)

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        raise NotImplementedError

    def train(self):
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        if load_policy is not None:
            self.policy = LOAD_POLICY()
            # import joblib
            # self.policy = joblib.load(self.load_policy)['policy']

        # initialize
        self.init_opt()
        # uninit_vars = []
        # for var in tf.global_variables():
        #     try:
        #         sess.run(var)
        #     except tf.errors.FailedPreconditionError:
        #         uninit_vars.append(var)
        # sess.run(tf.variables_initializer(uninit_vars))

        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            env = self.env
            while 'sample_goals' not in dir(env):
                env = env.wrapped_env

            learner_env_goals = env.sample_goals(self.meta_batch_size)

            self.policy.switch_to_init_dist()

            all_samples_data, all_paths = [], []
            for step in range(self.num_grad_updates + 1):
                paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix=str(step))
                all_paths.append(paths)
                samples_data = {}
                for key in paths.keys():
                    samples_data[key] = self.process_samples(itr, paths[key], log=False)
                all_samples_data.append(samples_data)

                self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True)
                self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))
                if step < self.num_grad_updates:
                    self.policy.compute_updated_dists(samples_data)

            self.optimize_policy(itr, all_samples_data)
            params = self.get_itr_snapshot(itr, all_samples_data[-1])
            if self.store_paths:
                params['paths'] = all_samples_data[-1]['paths']


class MAMLNPO(BatchMAMLPolopt):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.01,
                 use_maml=True,
                 **kwargs):
        assert optimizer is not None
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)

        if not use_maml:
            default_args = dict(batch_size=None, max_epochs=1)
            optimizer = FirstOrderOptimizer(**default_args)

        self.optimizer = optimizer
        self.step_size = step_size
        self.use_maml = use_maml
        self.kl_constrain_step = -1     # needs to be 0 or -1 (original pol params, or new pol params)
        super().__init__(**kwargs)

    def make_vars(self, stepnum='0'):
        pass

    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) == self.num_grad_updates + 1

        if not self.use_maml:
            all_samples_data = [all_samples_data[0]]

        input_list = []
        for step in range(len(all_samples_data)):
            obs_list, action_list, adv_list = [], [], []
            for i in range(self.meta_batch_size):
                inputs = ext.extract(all_samples_data[step][i], 'observations', 'actions', 'advantages')
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
            input_list += obs_list + action_list + adv_list     # [ [obs_0], [act_0], [adv_0], [obs_1], ... ]

            if step == 0:
                init_inputs = input_list

        if self.use_maml:
            dist_info_list = []
            for i in range(self.meta_batch_size):
                agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']
                dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

            input_list += tuple(dist_info_list)
            mean_kl_before = self.optimizer.constraint_val(input_list)

        loss_before = self.optimizer.loss(input_list)
        self.optimizer.optimize(input_list)
        loss_after = self.optimizer.loss(input_list)
        if self.use_maml:
            mean_kl = self.optimizer.constraint_val(input_list)

        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(itr=itr, policy=self.policy, baseline=self.baseline, env=self.env)













