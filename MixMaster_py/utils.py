import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(col=['종가']):
  """ Returns a 1 x n_step array """

  gold = pd.read_csv('../gold_daily_data/gold_krx_1000.csv', usecols=col)
  # recent price are at top; reverse it

  for c in col:
    gold[c] = gold[c].map(lambda x : int(x.replace(',','')))



  return np.array([ gold['종가'].values[::-1], ])

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