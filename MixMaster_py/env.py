import gym


class Env:
    env_state = {}

    def reset(self):
        pass


    def step(self, buy_num):
        # 스텝넘어가기
        # 액션이 구매할 주의 갯수로 표시됨.
        # state, reward, done(bool), _(기타 값 dictionary)
        pass

class Strategy:
    # 각자 알고리즘 트레이딩 구현
    pass


class Agent:
    # 전략을 모아서 결정을 내리는 agent
    strategy_list = []

    def __init__(self, strategy_list):
        self.strategy_list = strategy_list


if __name__ == "__main__":
    print("hoho")