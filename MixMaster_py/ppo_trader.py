from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from env.gymWrapper import create_gold_env

import os
import argparse

# issue 287
LOAD_DIR = os.path.join(os.getcwd(), "model")
SAVE_DIR = os.path.join(LOAD_DIR, "ppo_agent")


# Callback function printing episode statistics
def episode_finished(r):
    reward = "%.6f" % (r.episode_rewards[-1])
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=reward))

    if np.mean(r.episode_rewards[-1]) > 0 :
        r.agent.save_model(SAVE_DIR, append_timestep=False)
    return True

def print_simple_log(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))

def create_network_spec():
    network_spec = [
        {
            "type": "flatten"
        },
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='internal_lstm', size=32),
    ]
    return network_spec

def create_baseline_spec():
    baseline_spec = [
        {
            "type": "lstm",
            "size": 32,
        },
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
    ]
    return baseline_spec


def main(*args):
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=2000,
                      help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    args = parser.parse_args(args)



    # create environment for train and test
    PATH_TRAIN = "data/train/"
    PATH_TEST = "data/test/"
    TIMESTEP = 30 # window size
    environment = create_gold_env(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
    test_environment = create_gold_env(window_size=TIMESTEP, path=PATH_TEST, train=False)

    network_spec = create_network_spec()
    baseline_spec = create_baseline_spec()

    # TODO DQNAgent도 선택적 만들기.
    # TODO Agent Strategies 의존성을 UI에서 선택가능하게끔 변경해야함.


    agent = PPOAgent(
        discount=0.9999,
        states=environment.states,
        actions=environment.actions,
        network=network_spec,
        # Agent
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        # MemoryModel
        update_mode=dict(
            unit= 'timesteps', #'episodes',
            # 10 episodes per update
            batch_size= 32,
            # # Every 10 episodes
            frequency=10
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=50000
        ),
        # DistributionModel
        distributions=None,
        entropy_regularization=0.0,  # None
        # PGModel

        baseline_mode='states',
        baseline=dict(type='custom', network=baseline_spec),
        baseline_optimizer=dict(
            type='multi_step',
            optimizer=dict(
                type='adam',
                learning_rate=(1e-4)  # 3e-4
            ),
            num_steps=5
        ),
        gae_lambda=0,  # 0
        # PGLRModel
        likelihood_ratio_clipping=0.2,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=(1e-4)  # 1e-4
        ),
        subsampling_fraction=0.2,  # 0.1
        optimization_steps=10,
        execution=dict(
            type='single',
            session_config=None,
            distributed_spec=None
        )
    )

    train_runner = Runner(agent=agent, environment=environment)
    test_runner = Runner(
        agent=agent,
        environment=test_environment,
    )

    train_runner.run(episodes=5, max_episode_timesteps=16000, episode_finished=episode_finished)

    # TODO save models. UI쪽에서 사용할 메타데이터저장하기. https://tensorforce.readthedocs.io/en/latest/agents_models.html?highlight=agent
    # agent.save_model()

    # TODO load models.
    # agent.restore_model()

    # TODO TFTraderEnv에 에피소드마다의 포트폴리오 결과치 저장해야함. UI에 매순간 데이터 설정하기.
    # setResult(????)

    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=train_runner.episode,
        ar=np.mean(train_runner.episode_rewards[-100:]))
    )

    test_runner.run(num_episodes=1, deterministic=True, testing=True, episode_finished=print_simple_log)

if __name__ == '__main__':
    main()