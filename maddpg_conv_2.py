from util2 import make_env, plot_learning_curve

import torch as T
import cv2 as cv
import numpy as np
from maddpg.maddpg_conv import MADDPG
from maddpg.buffer_conv import MultiAgentReplayBuffer

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

def obs_list_to_state_vector(observation):
    state = np.array((3,84,84))
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def map_action(action):
    if action == 0:
        discrete_action = 'keep_lane'
    elif action == 1:
        discrete_action = 'slow_down'
    elif action == 2:
        discrete_action = 'change_lane_left'
    else:
        discrete_action = 'change_lane_right'
    return discrete_action

if __name__ == '__main__':
    scenario = '3line_intersection_2agents'
    n_agents = 2

    agent_spec = AgentSpec(interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=50))#, agent_builder=MADDPG)

    agent_specs = {'Agent-007': agent_spec, 'Agent-008': agent_spec}

    headless = True
    env = make_env('smarts.env:hiway-v0', agent_specs, 'scenarios/sumo/2agents_nohd', headless)

    actor_dims = env.observation_space.shape
    critic_dims = (3*n_agents, 84, 84)

    n_actions = 1
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, alpha=0.0001, beta=0.001, scenario=scenario, chkpt_dir='results/maddpg_conv/no_hd/')
    memory = MultiAgentReplayBuffer(100, actor_dims, n_actions, n_agents, batch_size=32)

    PRINT_INTERVAL = 100
    N_GAMES = 20000
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()
    
    agent_007_starting_pos = 'edge-west-WE'

    for i in range(N_GAMES):
        obs = env.reset()
        obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
        agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        while (len(obs.keys()) < 2) or (agent_007_starting_pos != agent_007_actual_pos):
            obs = env.reset()
            obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
            agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id

        score = 0
        dones = [False]*2
        episode_step = 0
        while not any(dones):
            actions = maddpg_agents.choose_action([obs['Agent-007'], obs['Agent-008']])
            discrete_action_1 = map_action(actions[0])
            discrete_action_2 = map_action(actions[1])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}

            obs_, rewards, dones, infos = env.step(agent_actions)

            dones = list(dones.values())
            dones = dones[:2]
            
            memory.store_transition([obs['Agent-007'], obs['Agent-008']], actions, rewards, [obs_['Agent-007'], obs_['Agent-008']], dones)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory, 32)

            obs = obs_

            score += sum(rewards)
            total_steps += 1
            episode_step += 1
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))        
