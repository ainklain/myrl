# from baseline import MyBaseline
from environment import PortfolioEnv
from policy import MyPolicy, MetaDDPG
# from model import MyModel
from sampler import EnvSampler
from ou import OrnsteinUhlenbeck
from memory import Memory

import numpy as np
import pandas as pd


class Argument:
    def __init__(self):
        self.dim_hidden_a = [64, 32, 16]
        self.dim_hidden_c = [32, 16]
        self.batch_size = 100
        self.max_path_length = 250
        self.n_itr = 10
        self.num_envs = 8
        self.M = 60
        self.K = 20
        self.gamma = 0.99


def main():
    args = Argument()

    env = PortfolioEnv()

    # policy = MyPolicy(env=env, dim_hidden=args.dim_hidden)

    model = MetaDDPG(env, gamma=args.gamma, dim_hidden_a=args.dim_hidden_a, dim_hidden_c=args.dim_hidden_c)

    memory = Memory(1000)

    sampler = EnvSampler(env, memory)

    for t in range(args.M + args.K, env.len_timeseries):
        if (t % 5 == 0) or (t < memory.memory_size):
            action_noise = OrnsteinUhlenbeck(mu=np.zeros(env.action_space.shape))
            policy_ori = policy.action_net.get_weights()
            tr_support = sampler.sample_trajectory(model, t - args.M, args.M, action_noise=action_noise)
            tr_target = sampler.sample_trajectory(model, t, args.K, action_noise=action_noise)

            memory.add(t - args.K)

        env_samples = sampler.sample_envs(args.num_envs)

        model.metatrain(env_samples)








        model.train()


    # baseline = MyBaseline(env_spec=env.spec)
    #
    # model = MyModel(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=args.batch_size,
    #     max_path_length=args.max_path_length,
    #     n_itr=args.n_itr)
    #




