from baseline import MyBaseline
from environment import PortfolioEnv
from policy import MyPolicy
from model import MyModel
from sampler import EnvSampler

import numpy as np
import pandas as pd
import tensorflow as tf


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
        name='policy',
        env_spec=env.spec,
        dim_hidden=args.dim_hidden)

    memory = MyMemory()

    sampler = EnvSampler(env, policy, memory)

    baseline = MyBaseline(env_spec=env.spec)

    model = MyModel(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=args.max_path_length,
        n_itr=args.n_itr)

    for t in range(env.len_timeseries):
        if t % 20 == 0:

        model.train()




