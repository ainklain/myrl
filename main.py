# from baseline import MyBaseline
from environment import PortfolioEnv
from policy import MyPolicy, MetaDDPG
# from model import MyModel
from sampler import EnvSampler
from ou import OrnsteinUhlenbeck
from memory import Memory

import numpy as np
import pandas as pd
import time


class Argument:
    def __init__(self):
        self.dim_hidden_a = [64, 32, 16]
        self.dim_hidden_c = [32]
        self.batch_size = 100
        self.max_path_length = 250
        self.n_itr = 10
        self.num_envs = 16
        self.M = 60
        self.K = 20
        self.gamma = 0.99


def main():
    args = Argument()

    env = PortfolioEnv()

    # policy = MyPolicy(env=env, dim_hidden=args.dim_hidden)


    memory = Memory(1000)

    sampler = EnvSampler(env, memory)
    model = MetaDDPG(sampler, args=args)

    rewards_list = []
    # for t in range(args.M + args.K, env.len_timeseries):
    for t in range(args.M + args.K, args.M + args.K + 501):
        s_t = time.time()
        if (t % 5 == 0) or (memory.memory_counter < memory.memory_size):
            print("{}. memory added".format(t))
            # action_noise = OrnsteinUhlenbeck(mu=np.zeros(env.action_space.shape))
            action_noise = None
            memory.add(t - args.K)

        # if memory.memory_counter < args.num_envs:
        #     print("{}. insufficient memory. learning skipped.".format(t))
        #     continue

        if t % args.K == 0:
            for _ in range(100):
                print("{}. sample environments.".format(t))
                env_samples = sampler.sample_envs(args.num_envs)
                print("{}. meta train start.".format(t))
                s_meta_t = time.time()
                model.metatrain(env_samples)
                e_meta_t = time.time()
                print("{}. fast adaptation.".format(t))
                s_fast_t = time.time()
                model.fast_train(t)
                e_fast_t = time.time()
                tr_real = model.get_trajectory(t, type_='target', training=False)
                rewards_list.append(tr_real)

        e_t = time.time()
        print("{}. total: {} / meta: {} / fast: {}".format(t, e_t - s_t, e_meta_t - s_meta_t, e_fast_t - s_fast_t))

    nav = np.zeros([len(rewards_list), args.K])
    winning_r = np.zeros_like(nav)
    instant_r = np.zeros_like(nav)
    cost = np.zeros_like(nav)
    for i in range(len(rewards_list)):
        for j in range(args.K):
            nav[i, j] = rewards_list[i][j]['info']['nav']
            winning_r[i, j] = rewards_list[i][j]['info']['winning_reward']
            instant_r[i, j] = rewards_list[i][j]['info']['instant_reward']
            cost[i, j] = rewards_list[i][j]['info']['costs']



    import pickle
    data = dict()
    data['nav'] = nav
    data['winning_r'] = winning_r
    data['instant_r'] = instant_r
    data['cost'] = cost
    f = open('data2.pkl', 'wb')
    pickle.dump(data, f)
    f.close()
