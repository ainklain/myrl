# from baseline import MyBaseline
from environment import PortfolioEnv
from policy import MyPolicy
# from model import MyModel
from sampler import EnvSampler
from ou import OrnsteinUhlenbeck
from memory import Memory

import numpy as np
import pandas as pd


class Argument:
    def __init__(self):
        self.dim_hidden = [64, 32, 16]
        self.batch_size = 100
        self.max_path_length = 250
        self.n_itr = 10


def main():
    args = Argument()

    env = PortfolioEnv()

    policy = MyPolicy(
        # name='policy',
        env=env,
        dim_hidden=args.dim_hidden)

    memory = Memory(100)

    action_noise = OrnsteinUhlenbeck(0)


    sampler = EnvSampler(env, memory)

    tr = sampler.sample_trajectory(policy, 0)

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
    # for t in range(env.len_timeseries):
    #     if t % 20 == 0:
    #
    #     model.train()




