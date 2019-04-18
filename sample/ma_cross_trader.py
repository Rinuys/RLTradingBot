import pandas as pd
import numpy as np
import talib


def get_dat(col='open'):
    samsung = pd.read_csv('../daily_data/cut_005930.csv', usecols=[col])
    naver = pd.read_csv('../daily_data/cut_035420.csv', usecols=[col])
    lg = pd.read_csv('../daily_data/cut_066570.csv', usecols=[col])

    stock_data = np.array([
        samsung[col].values[::-1].astype(float),
        naver[col].values[::-1].astype(float),
        lg[col].values[::-1].astype(float)
    ])
    return stock_data

def get_cross(lis1, lis2):
    result = [0, ]
    for i in range(len(lis1)-1):
        v = 0
        if np.NAN in (lis1[i], lis2[i], lis1[i+1], lis2[i+1]):
            pass
        elif (lis2[i]-lis1[i]) * (lis2[i+1]-lis1[i+1]) < 0:
            v = 1 if (lis2[i]-lis1[i]) < 0 else -1 # 1 : lis2 up, -1 : lis1 up
        result.append(v)
    return np.array(result)

def main(start=20000000, short_period=10, long_period=30):
    stock_data = get_dat()

    ma_short = [ talib.MA(lis, timeperiod=short_period) for lis in stock_data]
    ma_long = [ talib.MA(lis, timeperiod=long_period) for lis in stock_data]
    decision = np.array([
        get_cross(m10, m30) for m10, m30 in zip(ma_short, ma_long)
    ]) # -1 : sell, 1 : buy
    print("done calculatting to mmoving averge cross.")

    val_history = []
    num_stock = [0, 0, 0]
    cash_in_hand = start

    for time in range(len(stock_data[0])):
        today_close = stock_data[:,time]
        today_decision = decision[:,time]

        sell_index = [ i for i, d in enumerate(today_decision) if d==-1]
        buy_index = [i for i,d in enumerate(today_decision) if d==1]


        #sell
        for i in sell_index:
            cash_in_hand += num_stock[i]*today_close[i]
            num_stock[i]=0

        #buy
        can_buy = len(buy_index)>0
        while can_buy:
            for i in buy_index:
                if cash_in_hand > today_close[i]:
                    num_stock[i] += 1
                    cash_in_hand -= today_close[i]
                else:
                    can_buy = False

        #history write
        today_val = cash_in_hand + sum([ a*b for a,b in zip(num_stock, today_close) ])
        val_history.append( today_val )

        if (time+1)%10==0:
            print("episode and value %d : %d"%(time+1, today_val))

    with open("ma_trading_result_val.csv","wt") as f:
        columns = ["value"]
        print(*columns, sep=",", file=f)
        for row in val_history:
            print(row, sep=",", file=f)

if __name__=="__main__":
    main()