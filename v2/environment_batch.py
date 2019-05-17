
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns


def factor_history_csv():
    file_nm = 'data_for_metarl.csv'
    df = pd.read_csv(file_nm, index_col=0)

    df.columns = [i.lower() for i in df.columns]
    df = df[df.isna().sum(axis=1) == 0]
    # columns = list(df.columns)
    # marketdate = list(df.index.unique())
    # history = df.values

    return df   # , history, columns, marketdate


class PortfolioSim(object):
    def __init__(self,
                 asset_list,
                 macro_list=None,
                 trading_cost=1e-3):

        self.asset_list = asset_list
        self.trading_cost = trading_cost
        self.step_count = 0

        self.assets_return_df = pd.DataFrame(columns=asset_list)
        self.macros_return_df = None
        if macro_list is not None:
            self.macros_return_df = pd.DataFrame(columns=macro_list)

    def reset(self, n_envs, max_path_length=512):
        self.step_count = 0
        self.max_path_length = max_path_length
        self.navs = np.ones([n_envs, max_path_length])
        self.assets_nav = np.ones([n_envs, len(self.asset_list), max_path_length, ])
        self.rewards_history = np.ones(max_path_length)
        self.costs = np.zeros(max_path_length)

        self.last_pos = np.zeros([n_envs, len(self.asset_list)])
        self.dones = np.array([False for _ in range(n_envs)])

        self.assets_return_df = self.assets_return_df.iloc[0:0]
        if self.macros_return_df is not None:
            self.macros_return_df = self.macros_return_df.iloc[0:0]

    def _step(self, actions, assets_return, macros_return=None):
        eps = 1e-8
        self.assets_return_df.loc[self.step_count] = np.squeeze(assets_return)
        if macros_return is not None:
            self.macros_return_df.loc[self.step_count] = macros_return

        positions = ((assets_return + 1.) * actions) / (np.sum((assets_return + 1.) * actions, axis=1, keepdims=True) + eps)
        trades = actions - self.last_pos
        self.last_pos = positions

        trade_costs_pct = np.sum(abs(trades), axis=1, keepdims=True) * self.trading_cost
        costs = trade_costs_pct
        instant_reward = (np.sum((assets_return + 1.) * actions, axis=1, keepdims=True) - 1.) - costs

        # if self.step_count != 0:
        if self.step_count == 0:
            last_nav = np.ones([len(actions), 1])
            last_asset_nav = np.ones_like(actions)
        else:
            last_nav = self.navs[:, self.step_count - 1]
            last_asset_nav = self.assets_nav[:, :, self.step_count - 1]
        self.navs[:, self.step_count:(self.step_count+1)] = last_nav * (1. + instant_reward)
        self.assets_nav[:, :, self.step_count] = last_asset_nav * (1. + assets_return)

        winning_reward = np.zeros([len(actions), 1])
        for env_i in range(len(actions)):
            if self.dones[env_i]:
                continue
            if self.step_count == (self.max_path_length - 1):         # 최대 기간 도달
                self.dones[env_i] = True
                if self.navs[env_i, self.step_count] >= (1 + 0.05 * (self.step_count / 250)):      # 1년 5 % 이상 (목표)
                    winning_reward[env_i] = 100 * (1 + 0.05 * (self.step_count / 250))
                else:
                    winning_reward[env_i] = -1
            elif self.navs[env_i, self.step_count] >= (1 + 0.07 * (self.step_count / 250)):
                self.dones[env_i] = False
                winning_reward[env_i] = (self.navs[env_i, self.step_count] - (1 + 0.07 * (self.step_count / 250))) * 5
            elif self.navs[env_i, self.step_count] >= (1 + 0.05 * (self.step_count / 250)):
                self.dones[env_i] = False
                winning_reward[env_i] = (self.navs[env_i, self.step_count] - (1 + 0.05 * (self.step_count / 250))) * 2
            elif (self.navs[env_i, self.step_count] < np.max(self.navs[env_i, :]) * 0.5):        # MDD -50% 초과시
                self.dones[env_i] = True
                winning_reward[env_i] = -1
            elif (self.navs[env_i, self.step_count] < np.max(self.navs[env_i, :]) * 0.8):        # MDD -10% 초과시 패널티
                self.dones[env_i] = False
                winning_reward[env_i] = (self.navs[env_i, self.step_count] / (np.max(self.navs[env_i, :]) * 0.8) - 1.) * 10.
            elif (self.navs[self.step_count] < np.max(self.navs[env_i, :]) * 0.9):        # MDD -10% 초과시 패널티
                self.dones[env_i] = False
                winning_reward[env_i] = self.navs[env_i, self.step_count] / (np.max(self.navs[env_i, :]) * 0.9) - 1.
            elif (self.navs[self.step_count] < np.max(self.navs[env_i, :]) * 0.95):        # MDD -10% 초과시 패널티
                self.dones[env_i] = False
                winning_reward[env_i] = (self.navs[env_i, self.step_count] / (np.max(self.navs[env_i, :]) * 0.95) - 1.) * 0.3
            else:
                self.dones[env_i] = False
                winning_reward[env_i] = 0

        total_reward = instant_reward + winning_reward
        self.rewards_history[self.step_count] = total_reward

        info = {'instant_reward': deepcopy(instant_reward),
                'winning_reward': deepcopy(winning_reward),
                'nav': deepcopy(self.navs[self.step_count]),
                'costs': deepcopy(self.costs[self.step_count])}

        self.step_count += 1
        return total_reward, info, self.done

    def export(self):
        exported_data = dict()
        exported_data['last_step'] = self.step_count
        exported_data['asset_returns_df'] = self.assets_return_df[:self.step_count]
        exported_data['macro_returns_df'] = self.macros_return_df[:self.step_count]
        exported_data['navs'] = self.navs[:, self.step_count]
        exported_data['nav_returns'] = np.concatenate([[self.navs[:, 0] - 1.],
                                                       self.navs[:, 1:] / self.navs[:, -1] - 1.])[:self.step_count]
        exported_data['ew_nav'] = (1 + self.assets_return_df).cumprod(axis=0).mean(axis=1).values[:self.step_count]
        exported_data['ew_returns'] = np.concatenate([[exported_data['ew_nav'][0] - 1.], exported_data['ew_nav'][1:] / exported_data['ew_nav'][:-1] - 1.])[:self.step_count]
        exported_data['costs'] = self.costs[:self.step_count]
        exported_data['rewards_history'] = self.rewards_history[:self.step_count]

        return deepcopy(exported_data)


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_cost=0.0020,
                 input_window_length=250,
                 max_path_length=512,
                 is_training=True,
                 cash_asset=True):
        super().__init__()
        self.input_window_length = input_window_length
        self.max_path_length = max_path_length
        self.trading_cost = trading_cost
        self.cash_asset = cash_asset

        self.i_step = input_window_length
        self._setup()

    def _setup(self):
        df = factor_history_csv()
        asset_df = df[['mom', 'beme', 'gpa', 'kospi']]
        macro_df = df[['mkt_rf', 'smb', 'hml', 'rmw', 'wml', 'call_rate', 'usdkrw']]

        n_risky_asset = len(asset_df.columns)
        if self.cash_asset:
            self.asset_list = list(asset_df.columns) + ['cash']
        else:
            self.asset_list = list(asset_df.columns)

        if macro_df is not None:
            self.macro_list = list(macro_df.columns)
            assert asset_df.shape[0] == macro_df.shape[0], 'length of asset_df should be same as that of macro_df.'
        else:
            self.macro_list = []

        self.sim = PortfolioSim(self.asset_list,
                                self.macro_list,
                                max_path_length=self.max_path_length,
                                trading_cost=self.trading_cost)

        self.action_space = gym.spaces.Box(0.01, 0.99, shape=(len(self.asset_list), ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.input_window_length, n_risky_asset + len(self.macro_list), 3),
                                                dtype=np.float32)

        self._data = pd.concat([asset_df, macro_df], axis=1)

    def step(self, action, preprocess=True, debugging=False):
        assert len(action.shape) == 2, "action : [n_batch, action_dim], len of shape should be 2"
        return self._step(action, debugging=debugging)

    def _step(self, action, debugging=False):
        obs = self._data.iloc[(self.i_step - self.input_window_length):self.i_step]

        s = np.stack((obs.values,
                      self.preprocess(obs).values,
                      (obs.values >= 0) * 1. + (obs.values < 0) * -1.),
                     axis=-1)

        y1 = np.array(obs.iloc[-1][self.asset_list])

        if self.cash_asset:
            y1[-1] = 0

        y1 = y1.reshape([1, -1])
        reward, info, done2 = self.sim._step(action, y1, np.array(obs.iloc[-1][self.macro_list]))

        if debugging:
            print("{} reward: {} // y1: {} // info: {} // done: {}".format(
                self.sim.step_count, reward, y1, info, done2))
        self.i_step += 1
        if (self.i_step == self.len_timeseries) or done2:
            done = True
        else:
            done = False

        return s, reward, info, done

    def reset(self, t0=0, n_envs=1):
        return self._reset(t0, n_envs=n_envs)

    def _reset(self, t0=0, n_envs=1):
        self.i_step = self.input_window_length + t0
        self.sim.reset(n_envs)
        obs = self._data.iloc[(self.i_step - self.input_window_length):self.i_step]
        s = np.stack((obs.values,
                      self.preprocess(obs).values,
                      (obs.values >= 0) * 1. + (obs.values < 0) * -1.),
                     axis=-1)
        self.i_step += 1

        if n_envs > 1:
            self.render_call = -1
        else:
            self.render_call = 0

        return s

    def render(self, mode='human', close=False, statistics=False):
        return self._render(mode=mode,  close=close, statistics=statistics)

    def _render(self, mode='human', close=False, statistics=False):
        if mode == 'human':
            if self.render_call == -1:
                print("n_envs > 1. no rendering")
                return None

            if self.render_call == 0:
                # if hasattr(self, 'fig'):
                #     plt.close(self.fig)
                self.fig = plt.figure()
                self.ax1, self.ax2, self.ax3, self.ax4 = self.fig.subplots(4, 1)
                self.render_call += 1

            self._get_image(statistics)

            if self.render_call == 0:
                self.ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)
                self.render_call += 1
                self.ims = []

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _get_image(self, statistics=False):
        import io
        from PIL import Image

        render_data = self.sim.export()
        last_step = render_data['last_step']
        x_ = np.arange(last_step)
        self.ax1.plot(x_, render_data['navs'], color='k')
        self.ax1.plot(x_, render_data['ew_nav'], color='r')

        actions_t = np.transpose(render_data['actions'])
        pal = sns.color_palette("hls", 19)
        self.ax2.stackplot(x_, actions_t, colors=pal)

        asset_returns_df = render_data['asset_returns_df']
        asset_list = asset_returns_df.columns
        asset_cum_returns_t = np.cumprod(1 + np.transpose(asset_returns_df.values), axis=1)
        for i in range(len(asset_list)):
            self.ax3.plot(x_, asset_cum_returns_t[i], color=pal[i])

        self.ax4.plot(x_, render_data['rewards_history'])

        if statistics:
            mean_return = np.mean(render_data['nav_returns']) * 250
            std_return = np.std(render_data['nav_returns'], ddof=1) * np.sqrt(250)
            cum_return = render_data['navs'][-1] - 1
            total_cost = np.sum(render_data['costs'])

            ew_mean_return = np.mean(render_data['ew_returns']) * 250
            ew_std_return = np.std(render_data['ew_returns'], ddof=1) * np.sqrt(250)
            ew_cum_return = render_data['ew_nav'][-1] - 1

            max_nav = 1.
            max_nav_i = 0
            mdd_i = 0.
            mdd = list()
            for i in range(last_step):
                if render_data['navs'][i] >= max_nav:
                    max_nav = render_data['navs'][i]
                    max_nav_i = i
                else:
                    mdd_i = np.min(render_data['navs'][max_nav_i:(i+1)]) / max_nav - 1.
                mdd.append(mdd_i)
            max_mdd = np.min(mdd)

            print('model == ret:{} / std:{} / cum_return:{} / max_mdd:{} / cost:{}'.format(
                mean_return, std_return, cum_return, max_mdd, total_cost))
            print('ew_model == ret:{} / std:{} / cum_return:{}'.format(
                ew_mean_return, ew_std_return, ew_cum_return))






    def preprocess(self, obs):
        obs_p = (obs + 1).cumprod(axis=0) / (obs.iloc[0] + 1)
        obs_minmax = (obs_p - obs_p.min(axis=0)) / (obs_p.max(axis=0) - obs_p.min(axis=0))

        return obs_minmax

    @property
    def len_timeseries(self):
        return len(self._data.index) - self.input_window_length

    @property
    def step_count(self):
        return self.sim.step_count


