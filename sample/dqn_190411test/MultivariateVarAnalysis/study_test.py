# 다변량 var모형을 이용한 실험입니다.

from statsmodels.tsa.vector_ar.var_model import VARProcess
from  statsmodels.tsa.vector_ar.var_model import  VAR
import pickle
import pandas as pd
import matplotlib.pyplot as plt

pdframe = pd.DataFrame

with open("pickled_file.pkl",'rb') as f:
    dat = pickle.load(f)

diffs = dat[ ['close','open','high','low'] ].diff().dropna()
diffs.plot()
plt.show()