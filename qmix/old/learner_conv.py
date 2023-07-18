import numpy as np
import torch as T
import torch.optim as optim
from qmix.replay_memory_conv import MemoryBuffer
from qmix.networks_conv import CNNAgent, QMixer

class QMIX():
    def __init__(self, obs_dim, n_actions, n_agents, gamma=0.99, batch_size=32, mem_size=5000, eps=1.0, eps_min=0.05, lr=0.0005, replace=200, episode_length=50):
        self.obs_dim = obs_dim
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
        for i in range(n_agents):
            self.memory[i] = MemoryBuffer(self.mem_size, self.obs_dim, self.episode_length)
            self.agents_nets[i] = CNNAgent(self.obs_dim, self.n_actions, 'agent_' + str(i))
            self.target_agents_nets[i] = CNNAgent(self.obs_dim, self.n_actions, 'agent_' +str(i) + '_target')
            self.params += list(self.agents_nets[i].parameters())

        self.qmixer = QMixer(self.n_agents, 32, 'QMixer')
        self.target_qmixer = QMixer(self.n_agents, 32, 'QMixer_target')

        self.params += list(self.qmixer.parameters())
        self.optimizer = optim.RMSprop(params=self.params, lr=self.lr, alpha=0.99)

    def store_transition(self, agent_idx, ep, ep_step, obs, action, reward, obs_, done):
        self.memory[agent_idx].store_transition(ep, ep_step, obs, action, reward, obs_, done)

    def choose_actions(self, observations):
        actions = {}
        for i in range(self.n_agents):
            agent_obs = T.tensor([observations[i]], dtype=T.float)
            q_values = self.agents_nets[i].forward(agent_obs)
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
        for i in range(self.n_agents):
            obs[i], actions[i], rewards[i], obs_[i], dones[i] = self.memory[i].sample_buffer(self.batch_size)

            obs[i] = T.tensor(obs[i], dtype=T.float)#.reshape(self.obs_dim, 50*self.batch_size,1)
            actions[i] = T.tensor([actions[i]], dtype=T.float).reshape(1,50*self.batch_size,1)
            rewards[i] = T.tensor(rewards[i], dtype=T.float)
            obs_[i] = T.tensor(obs_[i], dtype=T.float)#.reshape(self.obs_dim, 50*self.batch_size,1)
            dones[i] = T.tensor(dones[i], dtype=bool)

        return obs, actions, rewards, obs_, dones

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

        obs, actions, rewards, obs_, dones = self.sample_memory()

        q_pred = T.tensor([])
        q_next = T.tensor([])
        indices = np.arange(self.batch_size*50)
        for i in range(self.n_agents):
            agent_q_pred = self.agents_nets[i].forward(obs[i].reshape(50*self.batch_size, 3, 84, 84))
            pippo = np.zeros(len(indices), dtype=np.int64)
            for j in range(len(indices)):
                pippo[j] =  actions[i][0][j].item()
            agent_q_pred = agent_q_pred[indices, pippo]
            agent_q_next = self.target_agents_nets[i].forward(obs_[i].reshape(50*self.batch_size, 3, 84, 84))
            agent_q_next = agent_q_next.max(dim=1)[0]
            q_pred = T.cat((q_pred, agent_q_pred))
            q_next = T.cat((q_next, agent_q_next))

        q_pred_tot = self.qmixer.forward(q_pred.reshape(50*self.batch_size,2))
        q_next_tot = self.target_qmixer.forward(q_next.reshape(50*self.batch_size,2))
        
        mask = dones[0] + dones[1]
        q_next_tot[mask.reshape(50*self.batch_size,1)] = 0.0
        rewards = rewards[0]
        rewards = rewards.reshape(50*self.batch_size, 1)
        q_target_tot = rewards + self.gamma*q_next_tot

        loss = ((q_target_tot - q_pred_tot)**2).sum()
        loss.backward()
        self.optimizer.step()

        for p in self.qmixer.parameters():
            p.data.clamp_(0)

        self.decrement_epsilon()
