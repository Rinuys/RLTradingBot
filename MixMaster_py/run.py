from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np

from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.execution import Runner
from env.gymWrapper import create_gold_env

import os
import argparse

# issue 287
LOAD_DIR = os.path.join(os.getcwd(), "model")
SAVE_DIR = os.path.join(LOAD_DIR, "trading_model")

def set_model_path(path):
    global LOAD_DIR
    global SAVE_DIR
    LOAD_DIR = path
    SAVE_DIR = os.path.join(LOAD_DIR, "trading_model")

gl_ui_window = None

# Callback function printing episode statistics
def episode_finished(r):
    # TODO 모델을 여기서 매호ㅓ 저장??
    reward = "%.6f" % (r.episode_rewards[-1])
    msg = "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=reward)
    print(msg)
    gl_ui_window.setInfo(msg=msg)
    gl_ui_window.portpolio_value_history.append(reward) # TODO 포트 폴리오 값 얻는법을 찾아야함.

    if np.mean(r.episode_rewards[-1]) > 0 :
        r.agent.save_model(SAVE_DIR, append_timestep=False)
    return True

def print_simple_log(r):
    msg = "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1])
    print(msg)
    gl_ui_window.setInfo(msg=msg)

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


def main(
        mode, # 'train'  or 'test'
        episode=2000,
        window_size=30, # agent 브레인이 참고할 이전 타임스텝의 길이
        init_invest=20000,
        model_path=None,
        addition_train=False,
        selected_learn='dqn', # 'dqn' or 'ppo'
        selected_trading=[],
        selected_subject=[],
        ui_windows=None, # 현재 띄워진 Ui객체
):
    global gl_ui_window
    gl_ui_window=ui_windows

    set_model_path(model_path if not model_path is None else os.path.join(os.getcwd(), 'model') )


    # create environment for train and test
    DATA_PATH='../daily_data'
    environment = create_gold_env(window_size=window_size, path=DATA_PATH, train=True if mode=='train' else False,
                                  selected_trading=selected_trading, selected_subject=selected_subject,
                                  init_invest=init_invest)


    network_spec = create_network_spec()
    baseline_spec = create_baseline_spec()

    if selected_learn=='ppo':
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
    else: # learn_model=='dqn' or etc.
        agent = DQNAgent(
            states=environment.states,
            actions=environment.actions,
            network=[
                dict(type='flatten'),
                dict(type='dense', size=32, activation='relu'),
                dict(type='dense', size=32, activation='relu'),
            ],
        )

    if mode=='test' or addition_train==True:
        if len([ elem for elem in os.listdir(LOAD_DIR) if 'trading_model' in elem ])>=3:
            agent.restore_model(LOAD_DIR)
            print('loaded')
        else:
            ui_windows.setInfo(msg="로딩할 트레이딩모델이 존재하지 않는 것으로 보입니다.")
            return

    runner = Runner(agent=agent, environment=environment)
    if mode=='train':
        kwargs=dict(
            episodes=episode, max_episode_timesteps=16000, episode_finished=episode_finished
        )
    else: # mode=='test'
        kwargs=dict(
            num_episodes=episode, deterministic=True, testing=True, episode_finished=print_simple_log
        )
    runner.run(**kwargs)

    # TODO save models. UI쪽에서 사용할 메타데이터저장하기. https://tensorforce.readthedocs.io/en/latest/agents_models.html?highlight=agent
    # agent.save_model()


    # TODO TFTraderEnv에 에피소드마다의 포트폴리오 결과치 저장해야함. UI에 매순간 데이터 설정하기.
    # setResult(????)
    msg = "{mode} finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        mode="Training" if mode=='train' else "Testing",
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:])
    )
    print(msg)
    ui_windows.setInfo(msg=msg)

if __name__ == '__main__':
    main()