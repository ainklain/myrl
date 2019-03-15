
import gym
import pandas as pd
import numpy as np


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
                 max_path_length=250,
                 trading_cost=1e-3):
        self.trading_cost = trading_cost
        self.max_path_length = max_path_length
        self.step_count = 0

        self.assets_return_df = pd.DataFrame(columns=asset_list)
        self.macros_return_df = None
        if macro_list is not None:
            self.macros_return_df = pd.DataFrame(columns=macro_list)

        self.actions = np.zeros([max_path_length, len(asset_list)])
        self.navs = np.ones(max_path_length)
        self.assets_nav = np.ones([max_path_length, len(asset_list)])
        self.positions = np.zeros([max_path_length, len(asset_list)])
        self.costs = np.zeros(max_path_length)
        self.trades = np.zeros([max_path_length, len(asset_list)])
        self.rewards_history = np.ones(max_path_length)

    def reset(self):
        self.step_count = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.assets_nav.fill(1)
        self.rewards_history.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)

        self.assets_return_df = self.assets_return_df.iloc[0:0]
        if self.macros_return_df is not None:
            self.macros_return_df = self.macros_return_df.iloc[0:0]

    def _step(self, actions, assets_return, macros_return=None):
        eps = 1e-8

        if self.step_count == 0:
            last_pos = np.zeros(len(actions))
            last_nav = 1.
            last_asset_nav = np.ones(len(actions))
        else:
            last_pos = self.positions[self.step_count - 1, :]
            last_nav = self.navs[self.step_count - 1]
            last_asset_nav = self.assets_nav[self.step - 1, :]

        self.assets_return_df.loc[self.step_count] = assets_return
        if macros_return is not None:
            self.macros_return_df.loc[self.step_count] = macros_return

        self.actions[self.step_count, :] = actions

        self.positions[self.step_count, :] = ((assets_return + 1.) * actions) / (np.dot((assets_return + 1.), actions) + eps)
        self.trades[self.step_count, :] = actions - last_pos

        trade_costs_pct = np.sum(abs(self.trades[self.step_count, :])) * self.trading_cost
        self.costs[self.step_count] = trade_costs_pct
        instant_reward = (np.dot((assets_return + 1.), actions) - 1.) - self.costs[self.step_count]

        if self.step_count != 0:
            self.navs[self.step_count] = last_nav * (1. + instant_reward)
            self.stk_nav[self.step_count, :] = last_asset_nav * (1. + assets_return)

        if (self.navs[self.step_count] == 0) | (self.navs[self.step_count] < np.max(self.navs) * 0.9):
            done = True
            winning_reward = -1
        elif self.step_count == self.steps:
            done = True
            if self.navs[self.step_count - 1] >= (1 + 0.05 * (self.step_count / 250)):      # 1년 5 % 이상 (목표)
                winning_reward = 1
            else:
                winning_reward = -1
        else:
            winning_reward = 0

        total_reward = 0.1 * instant_reward + 0.9 * winning_reward
        self.rewards_history[self.step_count] = total_reward

        info = {'instant_reward': instant_reward,
                'winning_reward': winning_reward,
                'nav': self.navs[self.step_count],
                'costs': self.costs[self.step_count]}

        self.step_count += 1
        return total_reward, info, done

    def export(self):
        exported_data = dict()
        exported_data['last_step'] = self.step_count
        exported_data['asset_returns_df'] = self.assets_return_df
        exported_data['macro_returns_df'] = self.macros_return_df
        exported_data['navs'] = self.navs
        exported_data['positions'] = self.positions
        exported_data['costs'] = self.costs
        exported_data['actions'] = self.actions

        return exported_data


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_cost=0.0020,
                 input_window_length=250,
                 max_path_length=20,
                 is_training=True):
        super().__init__()
        self.input_window_length = input_window_length
        self.max_path_length = max_path_length
        self.trading_cost = trading_cost

        self.i_step = input_window_length
        self._setup()

    def _setup(self):
        df = factor_history_csv()
        asset_df = df[['mom', 'beme', 'gpa', 'kospi']]
        macro_df = df[['mkt_rf', 'smb', 'hml', 'rmw', 'wml', 'call_rate', 'usdkrw']]

        self.asset_list = list(asset_df.columns)
        if macro_df is not None:
            self.macro_list = list(macro_df.columns)
            assert asset_df.shape[0] == macro_df.shape[0], 'length of asset_df should be same as that of macro_df.'
        else:
            self.macro_list = []

        self.sim = PortfolioSim(self.asset_list, self.macro_list, steps=250, trading_cost=self.trading_cost)

        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.asset_list), ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.input_window_length, len(self.asset_list) + len(self.macro_list)),
                                                dtype=np.float32)

        self._data = pd.concat([asset_df, macro_df], axis=1)

    def step(self, actions):
        return self._step(actions)

    def _step(self, actions, eps=1e-8):
        obs = self._data.iloc[(self.i_step - self.input_window_length):self.i_step]

        y1 = np.array(obs.iloc[-1][self.asset_list])
        reward, done2 = self.sim._step(actions, y1)

        self.i_step += 1
        self.step_count += 1
        if (self.step_count == self.max_path_length) or (self.i_step == self.len_timeseries) or done2:
            done = True
        else:
            done = False

        return obs, reward, done

    def reset(self, t0=0):
        return self._reset(t0)

    def _reset(self, t0=0):
        self.i_step = self.input_window_length + t0
        self.sim.reset()
        obs = self._data.iloc[(self.i_step - self.input_window_length):self.i_step]

        self.i_step += 1
        return obs

    def render(self, mode='human', close=False):
        return self._render(mode=mode,  close=close)

    def _render(self, mode='human', close=False):
        pass

    @property
    def len_timeseries(self):
        return len(self.df.index) - self.input_window_length

    @property
    def step_count(self):
        return self.sim.step_count

