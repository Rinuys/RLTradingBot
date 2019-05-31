import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


data_set_col = ['년/월/일','시가','종가','고가','저가']
data_set_index = '년/월/일'
data_set = {
  '금': pd.read_csv('../gold_daily_data/gold_krx_1000.csv', usecols=data_set_col, index_col=data_set_index),
  '석유': pd.read_csv('../gold_daily_data/fake_gold.csv', usecols=data_set_col, index_col=data_set_index),
}

# pd.DataFrame의 형태로된 종목 데이터들의 리스트가 반환됩니다.
def get_data(col=['종가'], select=None, mode='train', test_train_cut=0.75):
  data_cut_point = int(test_train_cut * len(data_set['금']))
  select = data_set.keys() if select is None else [select] # 하나의 종목코드 선택
  data_list = []

  for subject in select:
    dat = data_set[subject][col]
    dat_filter = pd.DataFrame()
    for c in col:
      # 최근 가격이 위쪽임 => 순서 역전시키기
      hoho = dat[c].map(lambda x : int(x.replace(',',''))).values[::-1]
      # 학습, 테스트 데이터 구분
      dat_filter[c] = hoho[:data_cut_point] if mode=='train' else hoho[data_cut_point:]
    data_list.append(dat_filter)
  return data_list

def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
    