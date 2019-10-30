import argparse
from env import Env
from model import DQN
import control as c
import time
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from collections import namedtuple
from tensorboardX import SummaryWriter
import os
import gc

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--simnum', type=int, default=0, metavar='N')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
parser.add_argument('--save-data', action='store_true', default=False)
parser.add_argument('--device', type=str, default="cpu", metavar='N')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N', help='log directory')
parser.add_argument('--data-directory', type=str, default='data/', metavar='N', help='data directory')
parser.add_argument('--epsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--wepsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--reward', type=float, default=1, metavar='N')
args = parser.parse_args()
torch.manual_seed(args.seed)


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
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self) :
        del self.memory
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

class Actor:
    def __init__(self, model_input_size, n_actions, device):
        self.epsilon = args.epsilon
        self.start_epoch = self.load_checkpoint()

        self.simnum = args.simnum
        self.log = args.log_directory + args.load_model + '/'
        self.writer = SummaryWriter(self.log + str(self.simnum) + '/')

        self.steps_done = 0
        self.n_actions = n_actions
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.trajectory = Trajectory(10000)
        self.replay_memory = ReplayMemory(10000)
        self.target_net = DQN(model_input_size, self.n_actions).to(self.device)

    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            if args.device == 'cpu':
                policy_net = torch.load(self.log + 'model.pt', map_location=lambda storage, loc: storage)
            else:
                policy_net = torch.load(self.log + 'model.pt')
            self.target_net.load_state_dict(policy_net['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), self.log + 'model.pt')

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, self.log + 'model.pt'))

    def save_checkpoint(self, idx):
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx + 1}
        torch.save(checkpoint, self.log + 'checkpoint{}.pt'.format(self.simnum))
        print('Actor {}: Checkpoint saved in '.format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))

    def load_checkpoint(self):
        if os.path.isfile(self.log + 'checkpoint{}.pt'.format(self.simnum)):
            checkpoint = torch.load(self.log + 'checkpoint{}.pt'.format(self.simnum))
            self.simnum = checkpoint['simnum']
            print("Actor {}: loaded checkpoint ".format(self.simnum), '(epoch {})'.format(checkpoint['epoch']), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return checkpoint['epoch']
        else:
            print("Actor {}: no checkpoint found at ".format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return args.start_epoch

    def save_memory(self, replay_memory):
        if os.path.isfile(self.log + 'memory.pt'):
            try:
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
            except:
                time.sleep(10)
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
        else:
            memory = {'replay_memory': self.replay_memory,
                      'priority': self.priority}
            torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
            self.replay_memory.clear()
            self.priority.clear()

        print('Actor {}: Memory saved in '.format(self.simnum), self.log + 'memory{}.pt'.format(self.simnum))

    def reward_function(self, trajectory) :
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        while ( sa_1 != None and sa_2 != None ) :
            # caculating reward
            reward = 0

            # push training data to replay memory
            torch_reward = torch.tensor([[reward]], device=self.device, dtype=torch.float)
            torch_action = torch.tensor([[sa_2.action]], device=self.device, dtype=torch.long)
            torch_state_1 = torch.tensor([sa_1.state], device=self.device, dtype=torch.float)
            torch_state_2 = torch.tensor([sa_2.state], device=self.device, dtype=torch.float)
            self.replay_memory.push(torch_state_2, torch_action, torch_state_1, torch_reward)

            sa_1 = sa_2
            sa_2 = trajectory.pop()

        self.save_memory(self.replay_memory)
        self.replay_memory.clear()
        self.trajectory.clear()

    def select_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 0.01
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.n_actions)

    def main(self):
        i_episode = self.start_epoch
        released = False
        pushed = False
        printed = False
        updated = False
        while True:
            # flag        // 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
            flag = env.flag()

            # observer
            if flag == 2 :
                state, got_state = env.state()

            # key press 
            if flag == 2 and got_state :
                action = self.select_action(state)
                env.step(action)
                released = False
            elif flag == 10 :
                pass                          # new cloud add here
            elif not released :
                env.release_key()
                released = True

            # trajectory push
            if flag == 2 and got_state :
                self.trajectory.push(state, action) # Store the state and action in trajectory
                pushed = False
            elif (flag == 3 or flag == 4) and not pushed : # if win or loss, Store the trajectory in replay memory
                self.reward_function(self.trajectory)
                pushed = True

            # printer
            if flag == 2 and got_state :
                print(state)
                printed = False
            elif flag == 3 and not printed :
                print("rest time")
                printed = True
            elif flag == 4 and not printed :
                print("gameset")
                printed = True

            # update
            if flag == 4 and not updated : 
                self.load_model()
                self.writer.add_scalar('total_reward', env.point(), i_episode)
                self.save_checkpoint(i_episode)
                i_episode += 1
                updated = True
            else :
                updated = False

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()
    actor = Actor(env.state_space_num, env.action_space_num, device)
    actor.main()
