import pickle

with open("/Users/jeonghan/Documents/RLTradingBot/sample/dqn_190411test/portfolio_val/201904111157-test.p","rb") as f:
    dat = pickle.load(f,encoding='latin1')

with open("/Users/jeonghan/Documents/RLTradingBot/sample/dqn_190411test/portfolio_val/test.csv","wt") as f:
    print(*dat, sep=", ", file=f)