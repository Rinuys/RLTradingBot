import process_data
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque
import env.Strategies as Strategies

class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, train=True, show_trade=True):
        self.train= train
        self.show_trade = show_trade
        self.path = path
        #self.actions = ["LONG", "SHORT", "FLAT"]
        self.n_strategies = len(Strategies.strategies)
        self.actions = dict(min_value=0.0, max_value=10.0, type="float", shape=(self.n_strategies,))
        self.fee = 0.0005
        self.seed()
        self.file_list = []
        # load_csv
        self.load_from_csv()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        #self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Box(low=0, high=10, shape=(self.n_strategies,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        raw_df= pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_frame(self, frame):
        offline_scaler = StandardScaler()
        observe = frame[..., :-4]
        observe = offline_scaler.fit_transform(observe)
        agent_state = frame[..., -4:]
        temp = np.concatenate((observe, agent_state), axis=1)
        return temp

    def step(self, action):
        s, r, d, i = self._step(action)
        self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue))), r, d, i

    def _step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        buy_value = [ w*strategy(self.closingPrices[:self.current_tick]) for strategy, w in zip(Strategies.strategies, action)]

        # two passes: sell first, then buy; might be naive in real-world settings
        buy_value = sum(buy_value)

        if v>3.0:
            # buy
            v = int(v)
            if v * self.closingPrice < self.cash_in_hand:
                pass
            else:
                v = self.cash_in_hand // self.closingPrice
            self.cash_in_hand -= int(self.closingPrice * v * (1.0 + self.fee))
            self.stock_owned += v

        elif v<-3.0:
            # sell
            v = int(abs(v))
            if self.stock_owned > v:
                pass
            else:
                v = self.stock_owned
            self.cash_in_hand += int(self.closingPrice * v * (1.0 - self.fee))
            self.stock_owned -= v
            
        else:
            # hold
            pass
        
        temp_portfolio = self.cash_in_hand + self.stock_owned * self.closingPrice
        self.portfolio = temp_portfolio
        self.reward += temp_portfolio - self.init_cash
        self.current_tick += 1

        if(self.show_trade and self.current_tick%100 == 0):
             print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))

        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.state = self.updateState()
        info = {'portfolio':np.array([self.portfolio]), "history":self.history}
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            if(self.train == False):
                np.array([info]).dump('info/ppo_{0}_LS.info'.format(self.portfolio))


        # # action comes from the agent
        # # 0 buy, 1 sell, 2 hold
        # # single position can be opened per trade
        # # valid action sequence would be
        # # LONG : buy - hold - hold - sell
        # # SHORT : sell - hold - hold - buy
        # # invalid action sequence is just considered hold
        # # (e.g.) "buy - buy" would be considred "buy - hold"
        # self.action = HOLD  # hold
        # if action == BUY: # buy
        #     if self.position == FLAT: # if previous position was flat
        #         self.position = LONG # update position to long
        #         self.action = BUY # record action as buy
        #         self.entry_price = self.closingPrice # maintain entry price
        #     elif self.position == SHORT: # if previous position was short
        #         self.position = FLAT  # update position to flat
        #         self.action = BUY # record action as buy
        #         self.exit_price = self.closingPrice
        #         self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1 # calculate reward
        #         self.krw_balance = self.krw_balance * (1.0 + self.reward) # evaluate cumulative return in krw-won
        #         self.entry_price = 0 # clear entry price
        #         self.n_short += 1 # record number of short
        # elif action == 1: # vice versa for short trade
        #     if self.position == FLAT:
        #         self.position = SHORT
        #         self.action = 1
        #         self.entry_price = self.closingPrice
        #     elif self.position == LONG:
        #         self.position = FLAT
        #         self.action = 1
        #         self.exit_price = self.closingPrice
        #         self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        #         self.krw_balance = self.krw_balance * (1.0 + self.reward)
        #         self.entry_price = 0
        #         self.n_long += 1

        # # [coin + krw_won] total value evaluated in krw won
        # if(self.position == LONG):
        #     temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        #     new_portfolio = self.krw_balance * (1.0 + temp_reward)
        # elif(self.position == SHORT):
        #     temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        #     new_portfolio = self.krw_balance * (1.0 + temp_reward)
        # else:
        #     temp_reward = 0
        #     new_portfolio = self.krw_balance

        # self.portfolio = new_portfolio
        # self.current_tick += 1
        # if(self.show_trade and self.current_tick%100 == 0):
        #     print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
        #     print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        
        return self.state, self.reward, self.done, info

    def reset(self):
        # hoho
        self.stock_owned = 0

        # self.current_tick = random.randint(0, self.df.shape[0]-800)
        if(self.train):
            self.current_tick = random.randint(0, self.df.shape[0] - 800)
        else:
            self.current_tick = 0

        print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_tick))

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.init_cash = 100 * 10000
        self.cash_in_hand = self.init_cash # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.closingPrice = self.closingPrices[self.current_tick]

        self.action = np.zeros(len(Strategies.strategies))
        self.done = False

        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()
        return self.state


    def preheat_queue(self):
        while(len(self.state_queue) < self.window_size):
            # rand_action = random.randint(0, len(self.actions)-1)
            # rand_action = 2
            rand_action = np.random.rand(len(Strategies.strategies))*10
            s, r, d, i= self._step(rand_action)
            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))

    def updateState(self):
        self.closingPrice = float(self.closingPrices[self.current_tick])
        state = self.df[self.current_tick]
        return state.reshape(1,-1)
