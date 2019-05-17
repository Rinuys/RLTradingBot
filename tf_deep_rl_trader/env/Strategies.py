# -1 : sell, 0 : hold, 1 : buy
def MovingAverage(close_history, long=30, short=5):
    if len(close_history)<long+1:
        return 0

    long_avg = sum(close_history[-long:]) / long
    short_avg = sum(close_history[-short:]) / short
    prev_long_avg = long_avg + (close_history[-long - 1] - close_history[-1]) / long
    prev_short_avg = short_avg + (close_history[-short - 1] - close_history[-1]) / short

    if (short_avg - long_avg) * (prev_short_avg - prev_long_avg) < 0:
        if short_avg > long_avg:
            return 1
        else:
            return -1
    return 0

strategies = [
    dict(function=MovingAverage,
         kwargs=dict(long=30,short=5),
         args=[]),
    dict(function=MovingAverage,
         kwargs=dict(long=60,short=10),
         args=[]),
]