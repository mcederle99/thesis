import numpy as np
import torch as T
import torch.optim as optim
from replay_memory import MemoryBuffer
from networks import RNNAgent, QMixer

class QMIX():
    def __init__(self, obs_dim, state_dim, n_actions, n_agents, gamma=0.99, batch_size=32, mem_size=5000, eps=1.0, eps_min=0.05, lr=0.0005, replace=200, episode_length=50):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = 0.00002
        self.lr = lr
        self.replace = replace
        self.episode_length = episode_length

        self.memory = {}
        self.agents_nets = {}
        self.target_agents_nets = {}
        self.params = []
        self.hidden_states = {}
        for i in range(n_agents):
            self.memory[i] = MemoryBuffer(self.mem_size, self.obs_dim, self.episode_length)
            self.agents_nets[i] = RNNAgent(self.obs_dim[0]+1, 64, self.n_actions, 'agent_' + str(i))
            self.hidden_states[i] = self.agents_nets[i].init_hidden()
            self.target_agents_nets[i] = RNNAgent(self.obs_dim[0]+1, 64, self.n_actions, 'agent_' +str(i) + '_target')
            self.params += list(self.agents_nets[i].parameters())

        self.qmixer = QMixer(self.n_agents, self.state_dim, 32, 'QMixer')
        self.target_qmixer = QMixer(self.n_agents, self.state_dim, 32, 'QMixer_target')

        self.params += list(self.qmixer.parameters())
        self.optimizer = optim.RMSprop(params=self.params, lr=self.lr, alpha=0.99)

    def store_transition(self, agent_idx, ep, ep_step, obs, action, reward, obs_, done, old_action):
        self.memory[agent_idx].store_transition(ep, ep_step, obs, action, reward, obs_, done, old_action)

    def choose_actions(self, old_actions, observations):
        actions = {}
        for i in range(self.n_agents):
            agent_obs = T.tensor(observations[i], dtype=T.float)
            old_acts = T.tensor([old_actions[i]], dtype=T.float)
            agent_obs = T.cat((agent_obs, old_acts)).reshape(-1, 10)
            q_values, self.hidden_states[i] = self.agents_nets[i].forward(agent_obs, self.hidden_states[i])
            if np.random.random() > self.eps:
                actions[i] = T.argmax(q_values).item()
            else:
                actions[i] = np.random.choice(self.n_actions)

        return actions

    def decrement_epsilon(self):
        if self.eps > self.eps_min:
            self.eps -= self.eps_dec
        else:
            self.eps = self.eps_min

    def replace_target_networks(self):
        self.target_qmixer.load_state_dict(self.qmixer.state_dict())
        for i in range(n_agents):
            self.target_agents_nets[i].load_state_dict(self.agents_nets[i].state_dict())

    def sample_memory(self):
        obs = {}
        actions = {}
        rewards = {}
        obs_ = {}
        dones = {}
        old_actions = {}
        for i in range(self.n_agents):
            obs[i], actions[i], rewards[i], obs_[i], dones[i], old_actions[i] = self.memory[i].sample_buffer(self.batch_size)

            obs[i] = T.tensor(obs[i], dtype=T.float).reshape(9,50*self.batch_size,1)
            actions[i] = T.tensor([actions[i]], dtype=T.float).reshape(1,50*self.batch_size,1)
            rewards[i] = T.tensor(rewards[i], dtype=T.float)
            obs_[i] = T.tensor(obs_[i], dtype=T.float).reshape(9,50*self.batch_size,1)
            dones[i] = T.tensor(dones[i], dtype=bool)
            old_actions[i] = T.tensor([old_actions[i]], dtype=T.float).reshape(1,50*self.batch_size,1)

        return obs, actions, rewards, obs_, dones, old_actions

    def save_models(self):
        self.qmixer.save_checkpoint()
        self.target_qmixer.save_checkpoint()
        for i in range(self.n_agents):
            self.agents_nets[i].save_checkpoint()
            self.target_agents_nets[i].save_checkpoint()

    def load_models(self):
        self.qmixer.load_checkpoint()
        self.target_qmixer.load_checkpoint()
        for i in range(self.n_agents):
            self.agents_nets[i].load_checkpoint()
            self.target_agents_nets[i].load_checkpoint()

    def learn(self):
        if self.memory[0].mem_cntr < self.batch_size:
            return
        self.optimizer.zero_grad()

        obs, actions, rewards, obs_, dones, old_actions = self.sample_memory()

        q_pred = T.tensor([])
        q_next = T.tensor([])
        indices = np.arange(self.batch_size*50)
        for i in range(self.n_agents):
            hidden_states = self.agents_nets[i].init_hidden()
            target_hidden_states = self.target_agents_nets[i].init_hidden()
            agent_q_pred, hidden_states = self.agents_nets[i].forward(T.cat((obs[i][:,0], old_actions[i][:,0])).reshape(1,10), hidden_states)
            agent_q_next, target_hidden_states = self.target_agents_nets[i].forward(T.cat((obs_[i][:,0], actions[i][:,0])).reshape(1,10), target_hidden_states) 
            for k in range(1, self.batch_size*50):
                tmp, hidden_states = self.agents_nets[i].forward(T.cat((obs[i][:,k], old_actions[i][:,k])).reshape(1,10), hidden_states)
                agent_q_pred = T.cat((agent_q_pred, tmp))
                tmp1, target_hidden_states = self.target_agents_nets[i].forward(T.cat((obs_[i][:,k], actions[i][:,k])).reshape(1,10), target_hidden_states)
                agent_q_next = T.cat((agent_q_next, tmp1))
            pippo = np.zeros(len(indices), dtype=np.int64)
            for j in range(len(indices)):
                pippo[j] =  actions[i][0][j].item()
            agent_q_pred = agent_q_pred[indices, pippo]
            agent_q_next = agent_q_next.max(dim=1)[0]
            q_pred = T.cat((q_pred, agent_q_pred))
            q_next = T.cat((q_next, agent_q_next))

        q_pred_tot = self.qmixer.forward(q_pred.reshape(50*self.batch_size,2), T.cat((obs[0], obs[1])))
        q_next_tot = self.target_qmixer.forward(q_next.reshape(50*self.batch_size,2), T.cat((obs_[0], obs_[1])))
        #print(q_pred_tot.shape) 
        mask = dones[0] + dones[1]
        q_next_tot[mask.reshape(50*self.batch_size,1)] = 0.0
        rewards = rewards[0]
        rewards = rewards.reshape(50*self.batch_size, 1)
        q_target_tot = rewards + self.gamma*q_next_tot

        loss = ((q_target_tot - q_pred_tot)**2).sum()
        loss.backward()
        self.optimizer.step()

        self.decrement_epsilon()
