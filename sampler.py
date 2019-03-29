import numpy as np


def dateadd(base_d, freq_='D', added_nums=0):
    # date_dt= datetime.datetime.strptime(base_d, '%Y-%m-%d')
    from dateutil.relativedelta import relativedelta
    from dateutil.parser import parse
    date_dt = parse(base_d)

    if freq_.lower() in ['y', 'year']:
        # date_dt = date_dt.replace(year=date_dt.year + added_nums)
        date_dt = date_dt + relativedelta(years=added_nums)
    elif freq_.lower() in ['m', 'month']:
        date_dt = date_dt + relativedelta(months=added_nums)
    elif freq_.lower() in ['w', 'week']:
        date_dt = date_dt + relativedelta(weeks=added_nums)
    else:
        date_dt = date_dt + relativedelta(days=added_nums)

    date_ = date_dt.strftime('%Y-%m-%d')
    return date_


class EnvSampler(object):
    def __init__(self, env, memory):
        self.env = env
        self.memory = memory

    def sample_trajectory(self, policy, t0, t_length, action_noise=None, training=True):
        done = False
        trajectory = list()
        obs = self.env.reset(t0)

        _i = 0
        while not done:

            action = policy.get_action(obs)
            if action_noise is not None and training is True:
                action = action + action_noise()
                action[action < 0.] = 0.0001
                action = action / np.sum(action)

            obs_, reward, info, done = self.env.step(action)

            # if training:
            #     trajectory.append({'obs': obs, 'action': action, 'reward': reward, 'obs_': obs_})
            # else:
            trajectory.append({'obs': obs, 'action': action, 'reward': reward, 'obs_': obs_, 'info': info})

            # print(_i, info, done)
            _i += 1
            obs = obs_.copy()
            if _i >= t_length:
                sim_data = self.env.sim.export()
                break

        return trajectory, sim_data

    def sample_envs(self, num_envs):
        return self.memory.sample_batch(num_envs)

    # def sample_envs(self, i_base_step, num_envs):
    #     idx_ = i_base_step + self.env.input_window_length - self.env.max_path_length
    #     envs = np.random.choice(np.arange(idx_), size=num_envs, replace=True)
    #     return envs

    def sample_envs_by_date(self, base_d, num_envs):
        date_ = dateadd(base_d, 'm', -1)
        idx_ = list(self.env._data.index).index(date_)

        # envs = np.random.choice(self.envs_list[:idx_], size=num_envs, replace=False)
        envs = np.random.choice(np.arange(idx_), size=num_envs, replace=True)
        return envs
