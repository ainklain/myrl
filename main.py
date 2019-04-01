# from baseline import MyBaseline
from environment import PortfolioEnv
from policy import MyPolicy, MetaDDPG, MetaDDPG2
# from model import MyModel
from sampler import EnvSampler
from ou import OrnsteinUhlenbeck
from memory import Memory

import numpy as np
import pandas as pd
import pickle
import time
import tensorflow as tf


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
        self.sampling_period = 5


def print_result(tr_data, t, j, t_length=20):
    tr_for_monitor, sim_monitor = tr_data
    print("t:{}-{} Total Rewards: {:.6f} (instant: {:.6f} / delayed: {:.6f})   Total Cost: {:.5f} bp".format(
        t, j,
        # tr_for_monitor[-1]['info']['instant_reward'] + tr_for_monitor[-1]['info']['winning_reward'],
        # tr_for_monitor[-1]['info']['instant_reward'],
        # tr_for_monitor[-1]['info']['winning_reward'],
        np.sum([tr_for_monitor[n]['info']['instant_reward'] + tr_for_monitor[n]['info']['winning_reward'] for n in range(len(tr_for_monitor))]),
        np.sum([tr_for_monitor[n]['info']['instant_reward'] for n in range(len(tr_for_monitor))]),
        np.sum([tr_for_monitor[n]['info']['winning_reward'] for n in range(len(tr_for_monitor))]),
        np.sum(sim_monitor['costs']) * 10000
        # tr_for_monitor[-1]['info']['costs'] * 10000
    ))
    print("t:{}-{} strategy: nav: {:.6f}, mean: {:.6f}({:.6f}) / std: {:.6f}({:.6f})\n".format(
        t, j,
        sim_monitor['navs'][t_length-1],
        np.mean(sim_monitor['nav_returns'][:t_length]),
        np.mean(sim_monitor['ew_returns'][:t_length]),
        np.std(sim_monitor['nav_returns'][:t_length], ddof=1),
        np.std(sim_monitor['ew_returns'][:t_length], ddof=1),
    ))


def main():
    args = Argument()

    env = PortfolioEnv()
    memory = Memory(100)

    sampler = EnvSampler(env, memory)
    model = MetaDDPG2(sampler, args=args)

    rewards_list = []
    sim_data = []

    T = 240
    pf_returns = np.zeros(T+1)
    pf_eq_returns = np.zeros(T+1)
    # for t in range(args.M + args.K, env.len_timeseries):
    n_cycle = 0
    t = args.M + args.K
    for t in range(args.M + args.K, args.M + args.K + memory.memory_size + T, args.sampling_period):
        assert args.K > args.sampling_period

        # collect samples
        action_noise = OrnsteinUhlenbeck(mu=np.zeros(env.action_space.shape))
        model.set_action_noise(action_noise)
        tr_support, tr_support_sim = model.get_trajectory(t - args.M, type_='support', training=True)
        memory.add([t, tr_support, tr_support_sim])
        print("{}. memory added".format(t))
        if memory.memory_counter < args.num_envs:
            continue

        if (t % args.K == 0):
            tr_support_test, _ = model.get_trajectory(t - args.M, type_='support', training=False)

            s_t = time.time()
            Js = 10000
            j_mod = 500
            j = 0
            for j in range(Js):
                env_samples = sampler.sample_envs(args.num_envs)
                s_meta_t = time.time()
                model.metatrain(env_samples)
                e_meta_t = time.time()

                if j % j_mod == 0:
                    # tr_data = model.get_trajectory(t, type_='target', training=False)

                    env_t = env_samples[0][0]
                    tr_support_print = model.get_trajectory(env_t - args.M, type_='support', training=False)
                    tr_target_print = model.get_trajectory(env_t, type_='target', training=False)
                    print('######################################')
                    print('test:')
                    print_result(tr_target_print, t, j, args.K)
                    print('after:')
                    print_result(tr_support_print, t, j, args.M)
                    print('before:')
                    print_result(env_samples[0][1:], t, j, args.M)


            for i in range(1000):
                model.fast_train(tr_support)
                # print(model.actor_net.get_weights()[0])
                # print(model.actor_net.get_weights()[-1])
                model.soft_update()
                if i % 100 == 0:
                    tr_support, tr_support_sim = model.get_trajectory(t - args.M, type_='support', training=True)
                    tr_support_print = model.get_trajectory(env_t - args.M, type_='support', training=False)
                    print_result(tr_support_print, t, i, args.M)
                    print_result([tr_support, tr_support_sim], t, i, args.M)


            model.save_optim_weight()
            s_fast_t = time.time()
            model.fast_train(tr_support_print[0])
            e_fast_t = time.time()
            print("t:{}. fast adaptation.".format(t))
            rewards_list.append(tr_real)
            sim_data.append(sim_)

            pf_returns[n_cycle:(n_cycle + args.K)] = sim_['nav_returns'][:args.K]
            pf_eq_returns[n_cycle:(n_cycle + args.K)] = sim_['ew_returns'][:args.K]
            n_cycle += 1

            e_t = time.time()
            print("{}. total: {} / meta: {} / fast: {}".format(t, e_t - s_t, e_meta_t - s_meta_t, e_fast_t - s_fast_t))

        f = open('model_w.pkl', 'wb')
        actor_w = model.actor_net.get_weights()
        critic_w = model.critic_net.get_weights()

        pickle.dump({'actor_w': actor_w, 'critic_w': critic_w}, f)

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



    data = dict()
    data['nav'] = nav
    data['winning_r'] = winning_r
    data['instant_r'] = instant_r
    data['cost'] = cost
    f = open('data2.pkl', 'wb')
    pickle.dump(data, f)
    f.close()


def main_onpolicy():
    args = Argument()

    env = PortfolioEnv()

    # policy = MyPolicy(env=env, dim_hidden=args.dim_hidden)

    memory = Memory(100)

    sampler = EnvSampler(env, memory)
    model = MetaDDPG(sampler, args=args)

    rewards_list = []
    sim_data = []

    T = 240
    pf_returns = np.zeros(T+1)
    pf_eq_returns = np.zeros(T+1)
    # for t in range(args.M + args.K, env.len_timeseries):
    n_cycle = 0
    for t in range(args.M + args.K, args.M + args.K + memory.memory_size + T):
        if (t % 5 == 0) or (memory.memory_counter < memory.memory_size):
            print("{}. memory added".format(t))
            # action_noise = None
            memory.add(t - args.K)
            if memory.memory_counter < memory.memory_size:
                continue

        # if memory.memory_counter < args.num_envs:
        #     print("{}. insufficient memory. learning skipped.".format(t))
        #     continue

        if t % args.K == 0:
            s_t = time.time()
            action_noise = OrnsteinUhlenbeck(mu=np.zeros(env.action_space.shape))
            model.set_action_noise(action_noise)
            if n_cycle == 0:
                Js = 100
                j_mod = 5
            else:
                Js = 40
                j_mod = 2
            for j in range(Js):
                # print("t:{}-{}. sample environments.".format(t, j))
                env_samples = sampler.sample_envs(args.num_envs)
                # print("t:{}-{}. meta train start.".format(t, j))
                s_meta_t = time.time()
                model.metatrain(env_samples)
                e_meta_t = time.time()

                if j % j_mod == 0:
                    tr_for_monitor, sim_monitor = model.get_trajectory(t, type_='target', training=False)
                    print("t:{}-{} Total Rewards: {:.6f} (instant: {:.6f} / delayed: {:.6f})   Total Cost: {:.5f} bp".format(t, j,
                        tr_for_monitor[-1]['info']['instant_reward'] + tr_for_monitor[-1]['info']['winning_reward'],
                        tr_for_monitor[-1]['info']['instant_reward'],
                        tr_for_monitor[-1]['info']['winning_reward'],
                        tr_for_monitor[-1]['info']['costs'] * 10000
                    ))
                    print("t:{}-{} strategy: mean: {:.6f}({:.6f}) / std: {:.6f}({:.6f})\n".format(t, j,
                        np.mean(sim_monitor['nav_returns'][:args.K]),
                        np.mean(sim_monitor['ew_returns'][:args.K]),
                        np.std(sim_monitor['nav_returns'][:args.K], ddof=1),
                        np.std(sim_monitor['ew_returns'][:args.K], ddof=1),
                    ))

            s_fast_t = time.time()
            model.fast_train(t)
            e_fast_t = time.time()
            print("t:{}. fast adaptation.".format(t))
            tr_real, sim_ = model.get_trajectory(t, type_='target', training=False)
            rewards_list.append(tr_real)
            sim_data.append(sim_)

            pf_returns[n_cycle:(n_cycle + args.K)] = sim_['nav_returns'][:args.K]
            pf_eq_returns[n_cycle:(n_cycle + args.K)] = sim_['ew_returns'][:args.K]
            n_cycle += 1

            e_t = time.time()
            print("{}. total: {} / meta: {} / fast: {}".format(t, e_t - s_t, e_meta_t - s_meta_t, e_fast_t - s_fast_t))

        f = open('model_w.pkl', 'wb')
        actor_w = model.actor_net.get_weights()
        critic_w = model.critic_net.get_weights()

        pickle.dump({'actor_w': actor_w, 'critic_w': critic_w}, f)

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



    data = dict()
    data['nav'] = nav
    data['winning_r'] = winning_r
    data['instant_r'] = instant_r
    data['cost'] = cost
    f = open('data2.pkl', 'wb')
    pickle.dump(data, f)
    f.close()
