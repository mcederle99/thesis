from util2 import make_env, plot_learning_curve

import torch as T
import cv2 as cv
import numpy as np
from maddpg1.maddpg_conv import MADDPG
from maddpg1.buffer_conv import MultiAgentReplayBuffer

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

def obs_list_to_state_vector(observation, env):
    stack = [collections.deque(maxlen=2)]

    for i in range(stack.maxlen):
        stack.append(observation[i])
    state = np.array(stack).reshape(env.observation_space.low.repeat(2,axis=0).shape)

    return state

def map_action(action):
    if action < -0.5:
        discrete_action = 'keep_lane'
    elif action < 0:
        discrete_action = 'slow_down'
    elif action < 0.5:
        discrete_action = 'change_lane_left'
    else:
        discrete_action = 'change_lane_right'
    return discrete_action

def evaluate(agents, env, ep, step, n_eval=3):
    score_history = []
    agent_007_starting_pos = 'edge-west-WE'
    obs = env.reset()
    obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
    agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
    while (len(obs.keys()) < 2) or (agent_007_starting_pos != agent_007_actual_pos):
        obs = env.reset()
        obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
        agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id

        score = 0
        dones = [False] * 2
        while not any(dones):
            actions = maddpg_agents.choose_action([obs['Agent-007'], obs['Agent-008']], evaluate=True)
            discrete_action_1 = map_action(actions['Agent-007'])
            discrete_action_2 = map_action(actions['Agent-008'])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}
            obs_, rewards, dones, infos = env.step(agent_actions)

            list_done = list(dones.values())
            dones = list_done[:2]

            obs = obs_
            score += sum(rewards)
        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f'Evaluation episode {ep} train steps {step}'
          f' average score {avg_score:.1f}')
    return avg_score


if __name__ == '__main__':
    scenario = '3line_intersection_2agents'
    n_agents = 2

    agent_spec = AgentSpec(interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=50))#, agent_builder=MADDPG)

    agent_specs = {'Agent-007': agent_spec, 'Agent-008': agent_spec}

    headless = True
    env = make_env('smarts.env:hiway-v0', agent_specs, 'scenarios/sumo/2agents_nohd', headless)

    actor_dims = [env.observation_space.shape, env.observation_space.shape]
    critic_dims = (env.observation_space.low.repeat(2,axis=0).shape)
    #critic_dims = (3*n_agents, 84, 84)

    n_actions = [1, 1]
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, alpha=0.0001, beta=0.001, env=scenario, gamma=0.95, chkpt_dir='results/maddpg1_conv/no_hd/')
    memory = MultiAgentReplayBuffer(20000, critic_dims, actor_dims, n_actions, n_agents, batch_size=32)

    EVAL_INTERVAL = 50
    MAX_STEPS = 1000000
    
    total_steps = 0
    episode = 0
    eval_steps = []
    eval_scores = []
    best_score = 0

    score = evaluate(maddpg_agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    agent_007_starting_pos = 'edge-west-WE'

    while total_steps < MAX_STEPS:
        obs = env.reset()
        obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
        agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id
        while (len(obs.keys()) < 2) or (agent_007_starting_pos != agent_007_actual_pos):
            obs = env.reset()
            obs, _, _, infos = env.step({'Agent-007': 'slow_down', 'Agent-008': 'slow_down'})
            agent_007_actual_pos = infos['Agent-007']['env_obs'][5].road_id

        dones = [False]*2
        while not any(dones):
            actions = maddpg_agents.choose_action([obs['Agent-007'], obs['Agent-008']])
            discrete_action_1 = map_action(actions['Agent-007'])
            discrete_action_2 = map_action(actions['Agent-008'])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}

            obs_, rewards, dones, infos = env.step(agent_actions)

            dones = list(dones.values())
            dones = dones[:2]
            list_obs = [obs['Agent-007'], obs['Agent-008']]
            list_obs_ = [obs_['Agent-007'], obs_['Agent-008']]
            state = obs_list_to_state_vector(list_obs, env)
            state_ = obs_list_to_state_vector(list_obs_, env)
            list_actions = list(actions.values())
            
            memory.store_transition(list_obs, state, list_actions, rewards, list_obs_, state_, dones)

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs = obs_

            total_steps += 1
        if episode % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)
            if score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = score
