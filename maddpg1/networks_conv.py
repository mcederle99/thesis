import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()) + 6)

    def forward(self, state, actions):
        layer1 = F.relu(self.conv1(state))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = layer3.view(layer3.size()[0], -1)
        layer4 = T.cat((layer4, actions), dim=1) 
        layer4 = F.relu(self.fc1(layer4))
        layer5 = F.relu(self.fc2(layer4))
        out = self.q(layer5)

        return out

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.pi = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer1 = F.relu(self.conv1(state))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = layer3.view(layer3.size()[0], -1)
        layer4 = F.relu(self.fc1(layer4))
        out = self.pi(layer4)

        return out

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkpt_file))
