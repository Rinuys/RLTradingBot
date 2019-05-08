# -1 : sell, 0 : hold, 1 : buy
def MovingAverage(price_history, long=30, short=5):
    if len(price_history)<long+1:
        return 0

    long_avg = sum(price_history[-long:])/long
    short_avg = sum(price_history[-short:])/short
    prev_long_avg = long_avg + (price_history[-long-1] - price_history[-1])/long
    prev_short_avg = short_avg + (price_history[-short-1] - price_history[-1])/short

    if (short_avg - long_avg) * (prev_short_avg - prev_long_avg) < 0:
        if short_avg > long_avg:
            return 1
        else:
            return -1
    return 0

strategies = [MovingAverage]