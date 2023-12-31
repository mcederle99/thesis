import numpy as np
import gym
from maddpg1.maddpg import MADDPG
from maddpg1.buffer import MultiAgentReplayBuffer

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def modify_obs(observations, agent_name):
    position = observations[agent_name][5][1]
    goal_position = observations[agent_name][5][10].goal.position
    rel_pos_to_goal = position - goal_position
    
    other_agent_position = observations[agent_name][7][0].position
    distance_bet_agents = position - other_agent_position
    
    hd_vehicle_position = observations[agent_name][7][1].position
    distance_from_hd = position - hd_vehicle_position
    
    heading_error = observations[agent_name][5][3].direction_vector()

    speed = observations[agent_name][5][4]

    steering = observations[agent_name][5][5]

    obs = [rel_pos_to_goal[0], rel_pos_to_goal[1], heading_error[0], heading_error[1], speed, steering, distance_bet_agents[0], distance_bet_agents[1], distance_from_hd[0], distance_from_hd[1]]

    return np.array(obs)

def modify_reward(rewards, observations, agent_name):
    reward = 0

    # Penalty for driving off road
    if observations[agent_name][4][1]:
        reward -= 10
        return np.float64(reward)

    # Penalty for driving on road shoulder
    if observations[agent_name][4][3]:
        reward -= 10
        return np.float64(reward)

    # Penalty for driving on wrong way
    if observations[agent_name][4][4]:
        reward -= 10
        return np.float64(reward)

    # Penalty for colliding
    if len(observations[agent_name][4][0]) != 0:
        reward -= 10
        return np.float64(reward)

    # Penalty for driving off route
    if observations[agent_name][4][2]:
        reward -= 10
        return np.float64(reward)

    # Bonus for reaching goal
    if observations[agent_name][4][6]:
        reward += 10
        return np.float64(reward)

    # Penalty for not moving
    #if observations[agent_name][4][5]:
    #    reward -= 5
    #    return np.float64(reward)

    reward += rewards[agent_name]
    return np.float64(reward)

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
    for i in range(n_eval):
        obs = env.reset()
        while (len(obs.keys()) < 2) or (agent_007_starting_pos != obs['Agent-007'][5][7]):
            obs = env.reset()

        score = 0
        dones = [False] * 2 
        obs_1 = modify_obs(obs, 'Agent-007')
        obs_2 = modify_obs(obs, 'Agent-008')
        while not any(dones):
            dict_obs = {'Agent-007': obs_1, 'Agent-008': obs_2}
            actions = maddpg_agents.choose_action(dict_obs, evaluate=True)
            discrete_action_1 = map_action(actions['Agent-007'])
            discrete_action_2 = map_action(actions['Agent-008'])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}
            obs_, rewards, dones, infos = env.step(agent_actions)

            obs_1_ = modify_obs(obs_, 'Agent-007')
            obs_2_ = modify_obs(obs_, 'Agent-008')

            reward_1 = modify_reward(rewards, obs_, 'Agent-007')
            reward_2 = modify_reward(rewards, obs_, 'Agent-008')
            list_reward = [reward_1, reward_2] 
            list_done = list(dones.values())
            dones = list_done[:2]

            obs_1 = obs_1_
            obs_2 = obs_2_
            score += sum(list_reward)
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

    env = gym.make('smarts.env:hiway-v0', 
                scenarios=['scenarios/sumo/2agents_hd'],
                agent_specs=agent_specs,
                #headless=False,
                #visdom=True
            )
    
    actor_dims = []
    n_actions = []
    for i in range(n_agents):
        actor_dims.append(10)
        n_actions.append(1)
    critic_dims = sum(actor_dims) + sum(n_actions)

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, env=scenario, gamma=0.95, alpha=0.0001, beta=0.001)
    critic_dims = sum(actor_dims)
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

    EVAL_INTERVAL = 50
    MAX_STEPS = 1000000

    total_steps = 0
    episode = 0
    best_score = 0
    eval_scores = []
    eval_steps = []
    test = True
    if test:
        while True:
            score = evaluate(maddpg_agents, env, episode, total_steps)
    score = evaluate(maddpg_agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    agent_007_starting_pos = 'edge-west-WE'

    while total_steps < MAX_STEPS:    
        obs = env.reset()
        while (len(obs.keys()) < 2) or (agent_007_starting_pos != obs['Agent-007'][5][7]):
            obs = env.reset()

        obs_1 = modify_obs(obs, 'Agent-007')
        obs_2 = modify_obs(obs, 'Agent-008')

        dones = [False]*n_agents
        while not any(dones):
            dict_obs = {'Agent-007': obs_1, 'Agent-008': obs_2}
            actions = maddpg_agents.choose_action(dict_obs)
            discrete_action_1 = map_action(actions['Agent-007'])
            discrete_action_2 = map_action(actions['Agent-008'])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}

            obs_, rewards, dones, infos = env.step(agent_actions)

            obs_1_ = modify_obs(obs_, 'Agent-007')
            obs_2_ = modify_obs(obs_, 'Agent-008')

            list_done = list(dones.values())
            list_obs = [obs_1, obs_2]
            list_actions = list(actions.values())
            list_obs_ = [obs_1_, obs_2_]
            
            state = obs_list_to_state_vector(list_obs)
            state_ = obs_list_to_state_vector(list_obs_)

            reward_1 = modify_reward(rewards, obs_, 'Agent-007')
            reward_2 = modify_reward(rewards, obs_, 'Agent-008')

           
            list_reward = [reward_1, reward_2] 
            dones = list_done[:2]
            
            memory.store_transition(list_obs, state, list_actions, list_reward, list_obs_, state_, dones)

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs_1 = obs_1_
            obs_2 = obs_2_

            total_steps += 1
        if episode % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, env, episode, total_steps)
            if score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = score
            eval_scores.append(score)
            eval_steps.append(total_steps)
            np.save('maddpg_2_scores.npy', np.array(eval_scores))
            np.save('maddpg_2_steps.npy', np.array(eval_steps))
        episode += 1
