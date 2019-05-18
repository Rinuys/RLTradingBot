from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import gym.wrappers
import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment
from env.TFTraderEnv import OhlcvEnv
import env.Strategies as Strategies

class OpenAIGym(Environment):
    """
    Bindings for OpenAIGym environment https://github.com/openai/gym
    To use install with "pip install gym".
    """

    def __init__(self, gym, monitor=None, monitor_safe=False, monitor_video=0, visualize=False):
        """
        Initialize OpenAI Gym.

        Args:
            gym_id: OpenAI Gym environment ID. See https://gym.openai.com/envs
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probabily going to slow down the training.
        """

        self.gym = gym
        self.visualize = visualize

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym = gym.wrappers.Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)

        self._states = dict(type='float', shape=gym.shape)
        self._actions = self.actions_ = dict(type='float', shape=len(Strategies.strategies))

    def __str__(self):
        return 'OpenAIGym({})'.format(self.gym.name)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def close(self):
        self.gym.close()
        self.gym = None

    def reset(self):
        if isinstance(self.gym, gym.wrappers.Monitor):
            self.gym.stats_recorder.done = True
        state = self.gym.reset()
        # return OpenAIGym.flatten_state(state=state)
        return state

    def execute(self, action):
        if self.visualize:
            self.gym.render()
        action = OpenAIGym.unflatten_action(action=action)
        state, reward, terminal, _ = self.gym.step(action)
        # return OpenAIGym.flatten_state(state=state), terminal, reward
        return state, terminal, reward

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                state = OpenAIGym.state_from_space(space=space)
                if 'type' in state:
                    states['state{}'.format(n)] = state
                else:
                    for name, state in state.items():
                        states['state{}-{}'.format(n, name)] = state
            return states
        elif isinstance(space, gym.spaces.Dict):
            states = dict()
            for space_name, space in space.spaces.items():
                state = OpenAIGym.state_from_space(space=space)
                if 'type' in state:
                    states[space_name] = state
                else:
                    for name, state in state.items():
                        states['{}-{}'.format(space_name, name)] = state
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def flatten_state(state):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['state{}-{}'.format(n, name)] = state
                else:
                    states['state{}'.format(n)] = state
            return states
        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states['{}'.format(state_name)] = state
            return states
        else:
            return state

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', num_actions=space.nvec[0], shape=num_discrete_space)
            else:
                actions = dict()
                for n in range(num_discrete_space):
                    actions['action{}'.format(n)] = dict(type='int', num_actions=space.nvec[n])
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['action{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                action = OpenAIGym.action_from_space(space=space)
                if 'type' in action:
                    actions['action{}'.format(n)] = action
                else:
                    for name, action in action.items():
                        actions['action{}-{}'.format(n, name)] = action
            return actions
        elif isinstance(space, gym.spaces.Dict):
            actions = dict()
            for space_name, space in space.spaces.items():
                action = OpenAIGym.action_from_space(space=space)
                if 'type' in action:
                    actions[space_name] = action
                else:
                    for name, action in action.items():
                        actions['{}-{}'.format(space_name, name)] = action
            return actions

        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def unflatten_action(action):
        if not isinstance(action, dict):
            return action
        elif all(
            name[:6] == 'action' and
            (name[6:name.index('-')].isnumeric() if '-' in name else name[6:].isnumeric())
            for name in action
        ):
            actions = list()
            n = 0
            while True:
                if any(name.startswith('action' + str(n) + '-') for name in action):
                    inner_action = {
                        name[name.index('-') + 1:] for name, inner_action in action.items()
                        if name.startswith('action' + str(n))
                    }
                    actions.append(OpenAIGym.unflatten_action(action=inner_action))
                elif any(name == 'action' + str(n) for name in action):
                    actions.append(action['action' + str(n)])
                else:
                    break
            return tuple(actions)
        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.unflatten_action(action=action)
            return actions

import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import process_data
import math, random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# def create_btc_env(window_size, path, train):
#     raw_env = OhlcvEnv(window_size=window_size, path=path, train=train)
#     env = OpenAIGym(raw_env, visualize=False)
#     return env

def create_gold_env(window_size, path, train):
    raw_env = OhlcvEnv(window_size=window_size, path=path, train=train)
    env = OpenAIGym(raw_env, visualize=False)
    return env

