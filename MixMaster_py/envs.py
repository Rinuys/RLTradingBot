import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools

from strategy import *

class TradingEnv(gym.Env):
  """
  A 3-stock (MSFT, IBM, QCOM) trading environment.

  State: [# of stock owned, current stock prices, cash in hand]
    - array of length n_stock * 2 + 1
    - price is discretized (to integer) to reduce state space
    - use close price for each stock
    - cash in hand is evaluated at each step based on action performed

  Action: sell (0), hold (1), and buy (2)
    - when selling, sell all the shares
    - when buying, buy as many as cash in hand allows
    - if buying multiple stock, equally distribute cash in hand and then utilize the balance

  Modified Action: use-w5 (5), use-w4 (4), use-w3 (3), use-w2 (2), use-w1 (1), hold(0),
    - Depends on each strategy.
    - when selling, sell all the shares
    - when buying, buy as many as cash in hand allows
    - if buying multiple stock, equally distribute cash in hand and then utilize the balance

  """
  def __init__(self, train_data, init_invest=20000, fee=0.0005):
    # data
    self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
    self.n_stock, self.n_step = self.stock_price_history.shape
    self.n_strategy = len(strategies)

    # instance attributes
    self.init_invest = init_invest
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None
    self.fee = fee

    # action space
    self.action_space = spaces.Box(0,10,(self.n_stock, self.n_strategy))

    # observation space: give estimates in order to sample and build scaler
    stock_max_price = self.stock_price_history.max(axis=1)
    stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
    price_range = [[0, mx] for mx in stock_max_price]
    cash_in_hand_range = [[0, init_invest * 2]]
    self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

    # seed and start
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = [0] * self.n_stock
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.cash_in_hand = self.init_invest
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    prev_val = self._get_val()
    self.cur_step += 1
    self.stock_price = self.stock_price_history[:, self.cur_step] # update price
    self._trade(action)
    cur_val = self._get_val()
    reward = cur_val - prev_val
    done = self.cur_step == self.n_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    obs.extend(self.stock_owned)
    obs.extend(list(self.stock_price))
    obs.append(self.cash_in_hand)
    return obs


  def _get_val(self):
    return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    # one pass to get buy_value per stock
    # applying strategy
    buy_value = []
    for strategy, weights in zip(strategies, action):
      strategy_invest = [ w * strategy(self.stock_price_history[i,:self.cur_step]) for i,w in enumerate(weights) ]
      buy_value.append(strategy_invest)

    buy_value = [ sum(v) for v in zip(*buy_value)]
    # two passes: sell first, then buy; might be naive in real-world settings
    for i,v in enumerate(buy_value):
      if v<0:
        sell_cnt = min(self.stock_owned[i], v)
        self.cash_in_hand += int(self.stock_price[i] * sell_cnt * (1.0 - self.fee))
        self.stock_owned[i] -= sell_cnt

    buy_index_value = [ (i,v) for i,v in enumerate(buy_value) if v>0 ]
    for i,v in buy_index_value:
      if self.cash_in_hand > self.stock_price[i] * v:
        self.stock_owned[i] += v
        self.cash_in_hand -= int(self.stock_price[i] * v * (1.0 + self.fee))


