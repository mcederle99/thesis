from util6 import make_env, plot_learning_curve

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

import torch as T
import numpy as np
import cv2 as cv
from dqn.dueling_ddqn_agent import DuelingDDQNAgent

if __name__ == '__main__':
    n_agents = 6

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        #agent_builder=SimpleAgent,
    )

    agent_specs = {
        "Agent-007": agent_spec,
        "Agent-008": agent_spec,
        "Agent-009": agent_spec
        "Agent-010": agent_spec,
        "Agent-011": agent_spec,
        "Agent-012": agent_spec
    }
    
    headless = False
    env = make_env("smarts.env:hiway-v0", agent_specs, 'scenarios/sumo/6agents_nohd', headless)
    
    agent_007 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')

    agent_008 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')
    
    agent_009 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')

    agent_010 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')

    agent_011 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')
    
    agent_012 = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=4, mem_size=20000, eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='results/dqn', algo='DuelingDDQNAgents', env_name='6agents_hd')

    best_score = 0
    load_checkpoint = False
    n_games = 10000

    if load_checkpoint:
        agent_007.load_models()
        agent_008.load_models()
        agent_009.load_models()
        agent_010.load_models()
        agent_011.load_models()
        agent_012.load_models()
    
    #fname = agent_007.algo + '_' + agent_007.env_name + '_lr' + str(agent_007.lr) + '_' + str(n_games) + 'games'
    #figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    #agent_007_starting_pos = 'edge-west-WE'
    #agent_008_starting_pos = 'edge-south-SN'
    action_dict = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

    for i in range(n_games):
        observations = env.reset()
        #_, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down', 'Agent-009': 'slow_down'})
        #agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        #agent_008_actual_pos = infos['Agent-008']['env_obs'][5].road_id
        while (len(observations.keys()) < 6):# or (agent_007_starting_pos != agent_007_actual_pos) or (agent_008_starting_pos != agent_008_actual_pos):
            observations = env.reset()
            #_, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down', 'Agent-009': 'slow_down'})
            #agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
            #agent_008_actual_pos = infos['Agent-008']['env_obs'][5].road_id
        dones = [False]*6
        score = 0
        while not any(dones):
            action_007 = agent_007.choose_action(observations['Agent-007'])
            action_008 = agent_008.choose_action(observations['Agent-008'])
            action_009 = agent_009.choose_action(observations['Agent-009'])
            action_010 = agent_007.choose_action(observations['Agent-010'])
            action_011 = agent_008.choose_action(observations['Agent-011'])
            action_012 = agent_009.choose_action(observations['Agent-012'])
            agent_actions = {'Agent-007': action_dict[action_007], 'Agent-008': action_dict[action_008], 'Agent-009': action_dict[action_009],'Agent-010': action_dict[action_010], 'Agent-011': action_dict[action_011], 'Agent-012': action_dict[action_012]}
            observations_, rewards, dones, infos = env.step(agent_actions)
            score += sum(rewards)
            dones = [dones['Agent-007'], dones['Agent-008'], dones['Agent-009'], dones['Agent-010'], dones['Agent-011'], dones['Agent-012']]
            
            if not load_checkpoint:
                agent_007.store_transition(observations['Agent-007'], action_007, rewards[0], observations_['Agent-007'], dones[0])
                agent_008.store_transition(observations['Agent-008'], action_008, rewards[1], observations_['Agent-008'], dones[1])
                agent_009.store_transition(observations['Agent-009'], action_009, rewards[2], observations_['Agent-009'], dones[2])
                agent_010.store_transition(observations['Agent-010'], action_010, rewards[3], observations_['Agent-010'], dones[3])
                agent_011.store_transition(observations['Agent-011'], action_011, rewards[4], observations_['Agent-011'], dones[4])
                agent_012.store_transition(observations['Agent-012'], action_012, rewards[5], observations_['Agent-012'], dones[5])

                agent_007.learn()
                #agent_008.learn()
                #agent_009.learn()
                #agent_010.learn()
                #agent_011.learn()
                #agent_012.learn()

            observations = observations_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %1.f best score %1.f epsilon %.2f' % (avg_score, best_score, agent_007.epsilon), 'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent_007.save_models()
                #agent_008.save_models()
                #agent_009.save_models()
            best_score = avg_score

        eps_history.append(agent_007.epsilon)
    #plot_learning_curve(steps_array, scores, eps_history, figure_file)
