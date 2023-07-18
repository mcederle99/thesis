import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import os

class CNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions, agent_id):
        super(CNNAgent, self).__init__()
        self.checkpoint_dir = 'results/qmix_conv/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, agent_id)
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64 ,64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_shape)

        self.fc4 = nn.Linear(fc_input_dims, 512)
        self.fc5 = nn.Linear(512, n_actions)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, inputs):
        layer1 = F.relu(self.conv1(inputs))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        # layer3 shape is BS x n_filters x H x W
        layer4 = layer3.view(layer3.size()[0], -1) # first dim is batch size, then -1 means that we flatten the other dims
        layer4 = F.relu(self.fc4(layer4))
        out = self.fc5(layer4)

        return out

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class QMixer(nn.Module):
    def __init__(self, n_agents, mixing_embed_dim, name):
        super(QMixer, self).__init__()
        self.checkpoint_dir = 'results/qmix_conv/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim
        self.input_dims = n_agents

        self.fc1 = nn.Linear(self.input_dims, mixing_embed_dim)
        self.fc2 = nn.Linear(mixing_embed_dim, 1)

    def forward(self, agent_qs):
        #bs = agent_qs.size(0)
        #agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        layer1 = F.relu(self.fc1(agent_qs))
        # Second layer
        q_tot = self.fc2(layer1)
        #q_tot = q_tot.view(bs, -1, 1)
        
        return q_tot

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
