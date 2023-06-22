import numpy as np
import gym
from qmix.learner import QMIX

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

def modify_obs(observations, agent_name):
    position = observations[agent_name][5][1]
    goal_position = observations[agent_name][5][10].goal.position
    rel_pos_to_goal = position - goal_position
    
    other_agent_position = observations[agent_name][7][0].position
    distance_bet_agents = position - other_agent_position

    heading_error = observations[agent_name][5][3].direction_vector()

    speed = observations[agent_name][5][4]

    steering = observations[agent_name][5][5]

    obs = [rel_pos_to_goal[0], rel_pos_to_goal[1], heading_error[0], heading_error[1], speed, steering, distance_bet_agents[0], distance_bet_agents[1]]

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
    #if observations[agent_name][4][6]:
    #    reward += 100
    #    return np.float64(reward)

    # Penalty for not moving
    #if observations[agent_name][4][5]:
    #    reward -= 5
    #    return np.float64(reward)

    reward += rewards[agent_name]
    return np.float64(reward)

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

    agent_spec = AgentSpec(interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=50))#, agent_builder=QMIX)

    agent_specs = {'Agent-007': agent_spec, 'Agent-008': agent_spec}

    env = gym.make('smarts.env:hiway-v0', 
                scenarios=['scenarios/sumo/2agents_nohd'],
                agent_specs=agent_specs,
                #headless=False,
                #visdom=True
            )
    
    n_actions = 4
    obs_dim = (8,)
    state_dim = (8*n_agents,1)
    qmix_agents = QMIX(obs_dim, state_dim, n_actions, n_agents, gamma=0.99, batch_size=2, mem_size=5000, eps=1.0, eps_min=0.05, lr=0.0005, replace=200, episode_length=50)

    PRINT_INTERVAL = 100
    N_GAMES = 20000
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        qmix_agents.load_models()
    
    agent_007_starting_pos = 'edge-west-WE' 

    for i in range(N_GAMES):
        observations = env.reset()
        while (len(observations.keys()) < 2) or (agent_007_starting_pos != observations['Agent-007'][5][7]):
            observations = env.reset()

        obs_1 = modify_obs(observations, 'Agent-007')
        obs_2 = modify_obs(observations, 'Agent-008')

        score = 0
        dones = [False]*2
        episode_step = 0
        old_actions = [0, 0]
        while not any(dones):
            actions = qmix_agents.choose_actions(old_actions, np.array([obs_1, obs_2]))
            discrete_action_1 = map_action(actions[0])
            discrete_action_2 = map_action(actions[1])
            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2}

            observations_, rewards, dones, infos = env.step(agent_actions)

            obs_1_ = modify_obs(observations_, 'Agent-007')
            obs_2_ = modify_obs(observations_, 'Agent-008')
            
            reward_1 = modify_reward(rewards, observations_, 'Agent-007')
            reward_2 = modify_reward(rewards, observations_, 'Agent-008')
            reward = reward_1 + reward_2

            dones = list(dones.values())
            dones = dones[:2]
            
            for j in range(n_agents):
                qmix_agents.store_transition(j, i, episode_step, obs_1, actions[j], reward, obs_1_, dones[j], old_actions[j])

            if not evaluate:
                if (i % 200 == 0) and (i > 0):
                    qmix_agents.replace_target_networks()
                qmix_agents.learn()

            obs_1 = obs_1_
            obs_2 = obs_2_
            old_actions = actions

            score += reward
            total_steps += 1
            episode_step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                qmix_agents.save_models()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))         
