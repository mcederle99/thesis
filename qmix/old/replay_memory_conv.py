import numpy as np

class MemoryBuffer():
    def __init__(self, mem_size, input_shape, episode_length):
        self.mem_size = mem_size
        self.episode_length = episode_length
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, episode_length, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, episode_length), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, episode_length), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, episode_length, *input_shape), dtype=np.float32)
        self.termination_memory = np.zeros((self.mem_size, episode_length), dtype=bool)
    
    def store_transition(self, ep, ep_step, obs, action, reward, obs_, done):
        ep_idx = ep % self.mem_size
        idx = ep_step
        
        self.state_memory[ep_idx][idx] = obs
        self.action_memory[ep_idx][idx] = action
        self.reward_memory[ep_idx][idx] = reward
        self.new_state_memory[ep_idx][idx] = obs_
        self.termination_memory[ep_idx][idx] = done
        
        self.mem_cntr = ep

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.termination_memory[batch]

        return states, actions, rewards, new_states, dones
