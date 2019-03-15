
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
    def __init__(self, env, policy, memory):
        self.env = env
        self.policy = policy
        self.memory = memory

    def sample_trajectory(self, policy, params=None, gamma=0.95):


    def sample_envs(self, base_d, num_envs):
        date_ = dateadd(base_d, 'm', -1)
        idx_ = self.envs_list.index(date_)
        # envs = random.sample(self.envs_list[:idx_], num_envs)
        envs = np.random.choice(self.envs_list[:idx_], size=num_envs, replace=False)
        return envs

