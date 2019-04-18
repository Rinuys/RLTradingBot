import pickle
#with open("portfolio_val/201904180240-train.p","rb") as fp:
with open("portfolio_val/201904180240-test.p","rb") as fp:
    a = pickle.load(fp)
    with open("output190418(test).csv","wt") as wf:
        print(a ,file=wf)
