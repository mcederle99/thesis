import gym
import sys
from util2 import make_env

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

import torch as T
import numpy as np
import cv2 as cv

action_dict = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

class SimpleAgent(Agent):
	def act(self, obs):
		return 'keep_lane'
        
#def modify_obs(observations, agent_name):
#	position = observations[agent_name][5][1]
#	goal_position = observations[agent_name][5][10].goal.position
#	goal_position = [goal_position[0], goal_position[1], goal_position[2]]
#	rel_pos_to_goal = np.array(position-goal_position)

#	heading_error = np.array(observations[agent_name][5][3].direction_vector())

#	speed = observations[agent_name][5][4]
#	steering = observations[agent_name][5][5]

	#neighborhood_vehicle_states = observations['Agent-007'][7]

#	new_obs = [rel_pos_to_goal[0], rel_pos_to_goal[1], rel_pos_to_goal[2], heading_error[0], heading_error[1], speed, steering]
#	new_obs = T.tensor([new_obs], dtype=T.float)
	
	
	
#	return new_obs
	
def modify_reward(rewards, observations, agent_name):
	reward = 0
	# Penalty for driving off road
	if observations[agent_name][4][1]:
	    reward -= 10
	    #print(f"ENV: Vehicle went off road.")
	    return np.float64(reward)

	# Penalty for driving on road shoulder
	if observations[agent_name][4][3]:
	    reward -= 10
	    #print(f"ENV: Vehicle went on road shoulder.")
	    return np.float64(reward)

	# Penalty for driving on wrong way
	if observations[agent_name][4][4]:
	    reward -= 10
	    #print(f"ENV: Vehicle went wrong way.")
	    return np.float64(reward)

	# Penalty for colliding
	if len(observations[agent_name][4][0]) != 0:
	    reward -= 10
	    #print(f"ENV: Vehicle collided.")
	    return np.float64(reward)

	# Penalty for driving off route
	if observations[agent_name][4][2]:
	    reward -= 10
	    #print(f"ENV: Vehicle went off route.")
	    return np.float64(reward)
	    
	# Bonus for reaching goal
	if observations[agent_name][4][6]:
	    reward += 10
	    #print(f"ENV: Vehicle reached goal.")
	    return np.float64(reward)
            
	reward += rewards[agent_name]
	return np.float64(reward)
	

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    agent_builder=SimpleAgent,
)

agent_specs = {
    "Agent-007": agent_spec,
    "Agent-008": agent_spec,
}

env = make_env("smarts.env:hiway-v0", agent_specs, 'scenarios/sumo/2agents_hd', False)

agents = {
    agent_id: agent_spec.build_agent()
    for agent_id, agent_spec in agent_specs.items()
}

i = 0
win_name = "camera"
while i<5:
    observations = env.reset()
    while len(observations.keys()) < 2:
	    observations = env.reset()
    #new_obs_1 = modify_obs(observations, "Agent-007")
    #new_obs_2 = modify_obs(observations, "Agent-008")
    dones = [False]*3
    while not any(dones):
        img = observations["Agent-008"]
        img = img.reshape((img.shape[1], img.shape[2], img.shape[0]))
        #cv.imshow(win_name, img)
        #cv.waitKey(0)
        agent_actions = {"Agent-007": agents["Agent-007"].act(observations['Agent-007']), "Agent-008": agents["Agent-008"].act(observations['Agent-007'])}
        observations, rewards, dones, infos = env.step(agent_actions)
        #print(rewards)
        #obs_1 = modify_obs(observations, "Agent-007")
	#obs_2 = modify_obs(observations, "Agent-008")
	#reward_1 = modify_reward(rewards, observations, "Agent-007")
	#reward_2 = modify_reward(rewards, observations, "Agent-008")
        dones = dones.values()
    i += 1
