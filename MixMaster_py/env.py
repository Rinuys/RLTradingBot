class BaseEnv:
    env_state = {}
    
    # 환경초기화
    def reset(self):
        pass

    # 스텝넘어가기
    # state, reward, done(bool), _(기타 값 dictionary)
    def step(self, action):
        pass