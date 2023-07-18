import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import os

class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions, agent_id):
        super(RNNAgent, self).__init__()
        self.checkpoint_dir = 'results/qmix/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, agent_id)

        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self, first_dim=1):
        return self.fc1.weight.new(first_dim, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in) 
        q = self.fc2(h)

        return q, h

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim, name):
        super(QMixer, self).__init__()
        self.checkpoint_dir = 'results/qmix/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim*self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = T.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(T.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = T.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = T.bmm(hidden, w_final) + v

        q_tot = y.view(bs, -1, 1)
        return q_tot

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
