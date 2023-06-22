from util2 import make_env, plot_learning_curve

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

import torch as T
import numpy as np
import cv2 as cv
from qmix.learner_conv import QMIX

if __name__ == '__main__':
    n_agents = 2

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=50),
        #agent_builder=SimpleAgent,
    )

    agent_specs = {
        "Agent-007": agent_spec,
        "Agent-008": agent_spec,
    }
    
    headless = False
    env = make_env("smarts.env:hiway-v0", agent_specs, 'scenarios/sumo/2agents_nohd', headless)
    qmix_agents = QMIX((env.observation_space.shape), 4, n_agents, gamma=0.99, batch_size=2, mem_size=100, eps=1.0, eps_min=0.05, lr=0.0005, replace=200, episode_length=50)

    best_score = 0
    load_checkpoint = False
    n_games = 20000

    if load_checkpoint:
        qmix_agents.load_models()
    
    fname = 'qmix_conv_2'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    agent_007_starting_pos = 'edge-west-WE'
    action_dict = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

    for i in range(n_games):
        observations = env.reset()
        observations, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
        agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        while (len(observations.keys()) < 2) or (agent_007_starting_pos != agent_007_actual_pos):
            observations = env.reset()
            observations, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
            agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        dones = [False]*2
        score = 0
        episode_step = 0
        while not any(dones):
            actions = qmix_agents.choose_actions([observations['Agent-007'], observations['Agent-008']])
            agent_actions = {'Agent-007': action_dict[actions[0]], 'Agent-008': action_dict[actions[1]]}
            observations_, rewards, dones, infos = env.step(agent_actions)
            reward = sum(rewards)
            score += sum(rewards)
            dones = [dones['Agent-007'], dones['Agent-008']]
            
            if not load_checkpoint:
                for j in range(n_agents):
                    qmix_agents.store_transition(j, i, episode_step, observations['Agent-007'], actions[0], reward, observations_['Agent-007'], dones[0])
                if (i % 200 == 0) and (i > 0):
                    qmix_agents.replace_target_networks
                qmix_agents.learn()

            observations = observations_
            n_steps += 1
            episode_step += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %1.f best score %1.f epsilon %.2f' % (avg_score, best_score, qmix_agents.eps), 'steps ', n_steps)

        if not load_checkpoint:
            if avg_score > best_score:
                qmix_agents.save_models()
                best_score = avg_score

        eps_history.append(qmix_agents.eps)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
