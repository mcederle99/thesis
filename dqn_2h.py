from util2 import make_env, plot_learning_curve

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

import torch as T
import numpy as np
import cv2 as cv
from dqn.dueling_ddqn_agent import DuelingDDQNAgent

if __name__ == '__main__':
    n_agents = 2

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        #agent_builder=SimpleAgent,
    )

    agent_specs = {
        "Agent-007": agent_spec,
        "Agent-008": agent_spec,
    }
    
    headless = True
    env = make_env("smarts.env:hiway-v0", agent_specs, 'scenarios/sumo/2agents_hd', headless)
    agent_007 = DuelingDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgent1', env_name='2agents_hd')

    agent_008 = DuelingDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgent2', env_name='2agents_hd')

    best_score = 0
    load_checkpoint = False
    TOTAL_STEPS = 305000

    if load_checkpoint:
        agent_007.load_models()
        agent_008.load_models()
    
    #fname = agent_007.algo + '_' + agent_007.env_name + '_lr' + str(agent_007.lr) + '_' + str(n_games) + 'games'
    #figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    n_episodes = 0
    time_to_eval = True
    scores, eval_scores, steps_array = [], [], []

    agent_007_starting_pos = 'edge-east-EW'
    action_dict = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

    while n_steps < TOTAL_STEPS:
        observations = env.reset()
        _, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
        agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        while (len(observations.keys()) < 2) or (agent_007_starting_pos != agent_007_actual_pos):
            observations = env.reset()
            _, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
            agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        dones = [False]*2
        score = 0
        while not any(dones):
            action_007 = agent_007.choose_action(observations['Agent-007'])
            action_008 = agent_008.choose_action(observations['Agent-008'])
            agent_actions = {'Agent-007': action_dict[action_007], 'Agent-008': action_dict[action_008]}
            observations_, rewards, dones, infos = env.step(agent_actions)
            score += sum(rewards)
            dones = [dones['Agent-007'], dones['Agent-008']]
            
            if not load_checkpoint:
                agent_007.store_transition(observations['Agent-007'], action_007, rewards[0], observations_['Agent-007'], dones[0])
                agent_008.store_transition(observations['Agent-008'], action_008, rewards[1], observations_['Agent-008'], dones[1])

                agent_007.learn()
                agent_008.learn()

            observations = observations_
            n_steps += 1
        
        n_episodes += 1
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        if time_to_eval:
            eval_scores.append(avg_score)
            steps_array.append(n_steps)
            time_to_eval = False
            print(avg_score)


        print('episode ', n_episodes, 'score: ', score, 'average score %1.f best score %1.f epsilon %.2f' % (avg_score, best_score, agent_007.epsilon), 'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent_007.save_models()
                agent_008.save_models()
            best_score = avg_score

        #eps_history.append(agent_007.epsilon)
    #plot_learning_curve(steps_array, scores, eps_history, figure_file)
