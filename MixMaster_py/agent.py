from collections import deque
import random
import numpy as np
from model import mlp
import itertools


class DQNAgent(object):
  """ A simple Deep Q agent """
  def __init__(self, state_size, action_size, invest_range=(0,10)):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.invest_range = invest_range
    self.invest_num = invest_range[1]-invest_range[0]+1
    self.model = mlp(state_size, self.invest_num*action_size[0]*action_size[1])



  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def act(self, state):
    # if np.random.rand() <= self.epsilon:
    #   return np.random.randint(-10,10,self.action_size)
    act_values = self.model.predict(state)
    act_values = act_values[0]

    invest_num = self.invest_num
    stock_num  = self.action_size[0]
    strategy_num = self.action_size[1]
    invest_list = []

    for i in range(0,len(act_values), invest_num):
      invest_list.append(np.argmax(act_values[i:i+invest_num]) - i - self.invest_range[0])

    invest_list_per_strategy = [ invest_list[v:v+stock_num] for v in range(0,len(invest_list), stock_num) ]

    return np.array(invest_list_per_strategy) # returns action


  def replay(self, batch_size=32):
    """ vectorized implementation; 30x speed up compared with for loop """
    minibatch = random.sample(self.memory, batch_size)

    # states = np.array([tup[0][0] for tup in minibatch])
    # actions = np.array([tup[1] for tup in minibatch])
    # rewards = np.array([tup[2] for tup in minibatch])
    # next_states = np.array([tup[3][0] for tup in minibatch])
    # done = np.array([tup[4] for tup in minibatch])

    states_all, actions, rewards, next_states_all, done = map(np.array, list(zip(*minibatch)))
    states = np.array(states_all[:,0])
    next_states = np.array(next_states_all[:,0])


    # Q(s', a)
    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]


    # Q(s, a)
    target_f = self.model.predict(states)
    # make the agent to approximately map the current state to future discounted reward
    target_f[range(batch_size), actions] = target

    self.model.fit(states, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)