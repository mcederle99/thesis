import numpy as np
import torch as T
import torch.nn.functional as F
from maddpg.agent_conv import Agent

def mapping(actions):
    new_actions = np.zeros(32, dtype=int)
    for i in range(32):
        new_actions[i] = actions[i][0]*4 + actions[i][1]
    return new_actions

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
            scenario='simple', alpha=0.01, beta=0.01,
            gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims, critic_dims,
                n_actions, agent_idx, alpha=alpha, beta=beta,
                chkpt_dir=chkpt_dir, n_agents=self.n_agents))

    def save_checkpoint(self):
        print(' ... saving checkpoint ... ')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print(' ... loading checkpoint ... ')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)

        return actions

    def learn(self, memory, batch_size):
        if not memory.ready():
            return

        actor_states, actions, rewards,\
                actor_new_states, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = np.array([actor_states[0], actor_states[1]]).reshape((batch_size, 3*self.n_agents, 84, 84))
        states_ = np.array([actor_new_states[0], actor_new_states[1]]).reshape((batch_size, 3*self.n_agents, 84, 84))
        
        states = T.tensor(states, dtype=T.float).to(device)    
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones)

        all_agents_new_actions =  []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                    dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[agent_idx],
                    dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
       
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_)
            new_actions1 = mapping(new_actions)
            tmp = T.tensor(np.zeros(32))
            for i in range(batch_size):
                tmp[i] = critic_value_[i][new_actions1[i]]
            critic_value_ = tmp
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states)
            old_actions1 = mapping(old_actions)
            tmp = T.tensor(np.zeros(32))
            for i in range(batch_size):
                tmp[i] = critic_value[i][old_actions1[i]]
            critic_value = tmp
            target = (rewards[:, agent_idx] + agent.gamma*critic_value_).flatten()
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states)
            mu1 = mapping(mu)
            tmp = T.tensor(np.zeros(32))
            for i in range(batch_size):
                tmp[i] = actor_loss[i][mu1[i]]
            actor_loss = tmp
            actor_loss = actor_loss.flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
