import pickle

with open("/Users/jeonghan/Documents/RLTradingBot/sample/dqn_190428_goldtest/portfolio_val/201904281053-test.p","rb") as f:
    dat = pickle.load(f,encoding='latin1')

with open("/Users/jeonghan/Documents/RLTradingBot/sample/dqn_190428_goldtest/portfolio_val/201904281053-test.csv","wt") as f:
    print(*dat, sep=", ", file=f)