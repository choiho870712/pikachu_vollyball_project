import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Transition_trajectory = namedtuple('Transition_trajectory', ('state', 'action'))
class Trajectory(object) :

    def __init__(self, capacity) :
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition_trajectory(*args)
            self.position = self.position + 1

    def pop(self) :
        if self.position > 0 :
            self.position = self.position - 1
            return self.memory[self.position]
        else :
            return None

    def clear(self) :
        del self.memory
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_size, 10)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.hidden_layer = nn.Linear(10, 10)
        self.hidden_layer.weight.data.normal_(0, 0.1)
        self.output_layer = nn.Linear(10, output_size)
        self.output_layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)

class Actor(object) :

    def __init__(self, model_input_size, n_actions, device) :
        self.steps_done = 0
        self.n_actions = n_actions
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.policy_net = DQN(model_input_size, self.n_actions).to(self.device)
        self.target_net = DQN(model_input_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.replay_memory = ReplayMemory(10000)

    def update_model(self) :
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_trajectory(self, trajectory, score) :
        # print("score = %d"%score)
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        reward_temp = 100
        while ( sa_1 != None and sa_2 != None ) :
            # caculating reward
            if ((sa_1.state[4] - 216) > 0) :
                reward_closedball = abs(sa_1.state[2] - sa_1.state[4]) / 36
            else :
                reward_closedball = 0
            
            if ( score == 1 ) :
                reward = reward_temp + reward_closedball
                reward_temp -= 20
            elif ( score == -1 ) :
                reward = -reward_temp + reward_closedball
                reward_temp -= 20
            else :
                print("error flag !")

            # push training data to replay memory
            torch_reward = torch.tensor([[reward]], device=self.device, dtype=torch.float)
            torch_action = torch.tensor([[sa_2.action]], device=self.device, dtype=torch.long)
            torch_state_1 = torch.tensor([sa_1.state], device=self.device, dtype=torch.float)
            torch_state_2 = torch.tensor([sa_2.state], device=self.device, dtype=torch.float)
            self.replay_memory.push(torch_state_2, torch_action, torch_state_1, torch_reward)

            sa_1 = sa_2
            sa_2 = trajectory.pop()

        trajectory.clear()

    def select_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 0.01
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return False
        transitions = self.replay_memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return True