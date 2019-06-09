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
import os

class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, selected_trading, selected_subject, train=True, show_trade=True,
                 init_invest=100*10000):
        self.init_invest=init_invest
        self.maxTrade = 1000.0
        self.train= train
        self.show_trade = show_trade
        self.holdFactor = 1.0
        self.path = path

        self.n_strategies = len(selected_trading)
        self.selected_trading = selected_trading
        self.subject_trade = selected_subject[0]
        self.subject_view = selected_subject[1]

        self.actions = dict(min_value=0.0, max_value=self.maxTrade, type="float", shape=(self.n_strategies,))
        self.fee = 0.01
        self.seed()
        self.subject_view_file_list = [ [] for sub in self.subject_view ]
        self.file_list = []
        # load_csv
        self.load_from_csv()

        # n_features
        self.window_size = window_size # agent 브레인이 참고할 이전 타임스텝의 길이
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, len(self.subject_view)*self.n_features)

        # defines action space
        #self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Box(low=0, high=int(self.maxTrade/2), shape=(self.n_strategies,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        '''
        subject_view의 순서대로
        self.subject_view_file_list, self.df_view 를 만듭니다.
        subject_trade의 self.df와 self.file_list도 생성합니다.
        :return:
        '''
        # 뷰잉으로 사용된 종목 데이터 로딩
        self.df_view = []
        for subject, file_list in zip(self.subject_view, self.subject_view_file_list):
            path = os.path.join(self.path, subject, 'train' if self.train else 'test')
            if(len(file_list) == 0):
                file_list = [x.name for x in Path(path).iterdir() if x.is_file()]
                file_list.sort()
            self.rand_episode = file_list.pop()
            raw_df= pd.read_csv(path + "/"+ self.rand_episode)
            extractor = process_data.FeatureExtractor(raw_df)
            df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

            ## selected manual fetuares
            feature_list = [
                'bar_hc',
                'bar_ho',
                'bar_hl',
                'bar_cl',
                'bar_ol',
                'bar_co', 'close']
            df.dropna(inplace=True) # drops Nan rows
            self.closingPrices = df['close'].values
            df = df[feature_list].values
            self.df_view.append(df)

        # 거래종목의 데이터 로딩
        file_list = self.file_list
        path = os.path.join(self.path, self.subject_trade, 'train' if self.train else 'test')
        if (len(file_list) == 0):
            file_list = [x.name for x in Path(path).iterdir() if x.is_file()]
            file_list.sort()
        self.rand_episode = file_list.pop()
        raw_df = pd.read_csv(path + "/" + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        df = extractor.add_bar_features()  # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        df.dropna(inplace=True)  # drops Nan rows
        self.closingPrices = df['close'].values
        self.df = df[feature_list].values

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

        buy_value = [ w*st['function'](self.closingPrices[:self.current_tick],*st['args'],**st['kwargs']) for st, w in zip(self.selected_trading, action)]

        # two passes: sell first, then buy; might be naive in real-world settings
        v = sum(buy_value)

        if v>1.0:
            # buy
            self.holdFactor = 1.0
            v = int(v)
            if v * self.closingPrice < self.cash_in_hand:
                pass
            else:
                v = self.cash_in_hand // self.closingPrice
            self.cash_in_hand -= int(self.closingPrice * v * (1.0 + self.fee))
            self.stock_owned += v

            self.tick_decision.append(1)
        elif v<-1.0:
            # sell
            self.holdFactor = 1.0
            v = int(abs(v))
            if self.stock_owned > v:
                pass
            else:
                v = self.stock_owned
            self.cash_in_hand += int(self.closingPrice * v * (1.0 - self.fee))
            self.stock_owned -= v

            self.tick_decision.append(-1)
            
        else:
            # hold
            self.holdFactor *= 1.01 # 음수에선 강한 부정리워드, 양수에선 강한 긍정리워드
            self.tick_decision.append(0)
            pass
        
        temp_portfolio = self.cash_in_hand + self.stock_owned * self.closingPrice
        self.portfolio = temp_portfolio
        self.reward += (temp_portfolio - self.init_invest) * self.holdFactor
        if(self.reward < -30):
            self.reward = -30
        if(self.reward > 30):
            self.reward = 30
        self.current_tick += 1

        self.tick_value.append(temp_portfolio) # tick마다 포트폴리오 가치 저장

        if(self.show_trade and self.current_tick%100 == 0):
             print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))

        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.state = self.updateState()
        info = {'portfolio':np.array([self.portfolio]), "history":self.history}
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
        if (self.current_tick >= self.end_tick - self.window_size-1):
            self.done = True
            # if(self.train == False):
            #     np.array([info]).dump('info/ppo_{0}_LS.info'.format(self.portfolio))
        
        return self.state, self.reward, self.done, info

    def reset(self):
        # hoho
        self.stock_owned = 0
        self.holdFactor = 1.0

        if(self.train):
            self.current_tick = random.randint(0, self.df.shape[0] - 800)
        else:
            self.current_tick = 0

        self.end_tick = random.randint(self.current_tick, self.df.shape[0] - self.window_size + 1)

        print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_tick))

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.cash_in_hand = self.init_invest # initial balance, u can change it to whatever u like
        self.portfolio = float(self.cash_in_hand) # (coin * current_price + current_krw_balance) == portfolio
        self.closingPrice = self.closingPrices[self.current_tick]
        #self.current_tick = 0
        self.action = np.zeros(self.n_strategies)
        self.done = False

        # 테스트 틱데이터 초기화
        self.tick_value = []
        self.tick_decision = []

        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()
        return self.state


    def preheat_queue(self):
        while(len(self.state_queue) < self.window_size):
            # rand_action = random.randint(0, len(self.actions)-1)
            # rand_action = 2
            rand_action = np.random.rand(self.n_strategies)*self.maxTrade
            s, r, d, i= self._step(rand_action)
            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))

    def updateState(self):
        self.closingPrice = float(self.closingPrices[self.current_tick])

        serialize = np.concatenate([ df[self.current_tick] for df in self.df_view])
        return serialize.reshape(1,-1)
