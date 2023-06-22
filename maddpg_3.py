import numpy as np
import gym
from maddpg.maddpg import MADDPG
from maddpg.buffer import MultiAgentReplayBuffer

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def modify_obs(observations, agent_name, other_agents_name):
    position = observations[agent_name][5][1]
    goal_position = observations[agent_name][5][10].goal.position
    rel_pos_to_goal = position - goal_position
    
    agent1_position = observations[other_agents_name[0]][5][1]
    distance_from_agent1 = position - agent1_position
    #distance_from_agent1 = np.sqrt(distance_from_agent1[0]**2 + distance_from_agent1[1]**2 + distance_from_agent1[2]**2)
    agent2_position = observations[other_agents_name[1]][5][1]
    distance_from_agent2 = position - agent2_position
    #distance_from_agent2 = np.sqrt(distance_from_agent2[0]**2 + distance_from_agent2[1]**2 + distance_from_agent2[2]**2)

    heading_error = observations[agent_name][5][3].direction_vector()

    speed = observations[agent_name][5][4]

    steering = observations[agent_name][5][5]

    obs = [rel_pos_to_goal[0], rel_pos_to_goal[1], heading_error[0], heading_error[1], speed, steering, distance_from_agent1[0], distance_from_agent1[1], distance_from_agent2[0], distance_from_agent2[1]]

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
    #    reward += 500
    #     return np.float64(reward)

    # Penalty for reaching max number of steps
    #if observations[agent_name][4][7]:
    #    reward -= 50
    #    return np.float64(reward)

    reward += rewards[agent_name]
    return np.float64(reward)

def map_action(action, eps):
    action_dict = {0: 'keep_lane', 1: 'slow_down'}
    prob = np.random.rand(1)
    if action < 0.5:
        discrete_action = 0
    else:
        discrete_action = 1
    if prob >= eps:
        return action_dict[discrete_action]
    final_action = np.random.choice(list(action_dict.values()))
    return final_action

if __name__ == '__main__':
    scenario = '3line_intersection_3agents'
    n_agents = 3

    agent_spec = AgentSpec(interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=70))#, agent_builder=MADDPG)

    agent_specs = {'Agent-007': agent_spec, 'Agent-008': agent_spec, 'Agent-009': agent_spec}

    env = gym.make('smarts.env:hiway-v0', 
                scenarios=['scenarios/sumo/3agents_nohd'],
                agent_specs=agent_specs,
                headless=False,
                #visdom=True
            )
    
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(10)
    critic_dims = sum(actor_dims)

    n_actions = 1
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.0001, beta=0.001, scenario=scenario, chkpt_dir='results/maddpg/no_hd/')
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

    action_dict = {'keep_lane': 0, 'slow_down': 1}

    PRINT_INTERVAL = 100
    N_GAMES = 20000
    total_steps = 0
    score_history = []
    evaluate = True
    best_score = -200

    if evaluate:
        maddpg_agents.load_checkpoint()
        eps = 0
    else:
        eps = 0.9

    observations = env.reset()
    agent_007_starting_pos = observations['Agent-007'][5][7]
    agent_008_starting_pos = observations['Agent-008'][5][7]

    exploration_cntr = 0
    for i in range(N_GAMES):
        observations = env.reset()
        if not evaluate:
            while (len(observations.keys()) < 3) or (agent_007_starting_pos != observations['Agent-007'][5][7]) or (agent_008_starting_pos != observations['Agent-008'][5][7]):
                observations = env.reset()

        obs_1 = modify_obs(observations, 'Agent-007', ['Agent-008', 'Agent-009'])
        obs_2 = modify_obs(observations, 'Agent-008', ['Agent-007', 'Agent-009'])
        obs_3 = modify_obs(observations, 'Agent-009', ['Agent-007', 'Agent-008'])

        score = 0
        dones = [False]*3
        episode_step = 0
        while not any(dones):
            actions = maddpg_agents.choose_action(np.array([obs_1, obs_2, obs_3]))
            discrete_action_1 = map_action(actions[0], eps)
            discrete_action_2 = map_action(actions[1], eps)
            discrete_action_3 = map_action(actions[2], eps)

            actions = [action_dict[discrete_action_1], action_dict[discrete_action_2], action_dict[discrete_action_3]]

            agent_actions = {'Agent-007': discrete_action_1, 'Agent-008': discrete_action_2, 'Agent-009': discrete_action_3}

            observations_, rewards, dones, infos = env.step(agent_actions)

            obs_1_ = modify_obs(observations_, 'Agent-007', ['Agent-008', 'Agent-009'])
            obs_2_ = modify_obs(observations_, 'Agent-008', ['Agent-007', 'Agent-009'])
            obs_3_ = modify_obs(observations_, 'Agent-009', ['Agent-007', 'Agent-008'])

           
            state = obs_list_to_state_vector([obs_1, obs_2, obs_3])
            state_ = obs_list_to_state_vector([obs_1_, obs_2_, obs_3_])

            reward_1 = modify_reward(rewards, observations_, 'Agent-007')
            reward_2 = modify_reward(rewards, observations_, 'Agent-008')
            reward_3 = modify_reward(rewards, observations_, 'Agent-009')

            dones = list(dones.values())
            dones = dones[:3]
            
            memory.store_transition([obs_1, obs_2, obs_3], state, actions, [reward_1, reward_2, reward_3], [obs_1_, obs_2_, obs_3_], state_, dones)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs_1 = obs_1_
            obs_2 = obs_2_
            obs_3 = obs_3_

            score += sum([reward_1, reward_2, reward_3])
            total_steps += 1
            episode_step += 1

        exploration_cntr += 1
        if exploration_cntr == 1000:
            exploration_cntr = 0
            eps = eps/1.1
        if not evaluate:
            if eps < 0.1:
                eps = 0.1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
