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
parser.add_argument('--history_size', type=int, default=4, metavar='N')
parser.add_argument('--hidden-size', type=int, default=32, metavar='N')
parser.add_argument('--epsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--wepsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--reward', type=float, default=1, metavar='N')
parser.add_argument('--replay-size', type=int, default=3000, metavar='N')
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


class Actor:
    def __init__(self):
        if args.device != 'cpu':
            torch.cuda.set_device(int(args.device))
            self.device = torch.device('cuda:{}'.format(int(args.device)))
        else:
            self.device = torch.device('cpu')

        self.simnum = args.simnum
        self.history_size = args.history_size
        self.hidden_size = args.hidden_size

        self.epsilon = args.epsilon
        self.log = args.log_directory + args.load_model + '/'
        self.writer = SummaryWriter(self.log + str(self.simnum) + '/')

        self.dis = 0.99
        self.win = False

        self.replay_memory = deque(maxlen=args.replay_size)
        self.priority = deque(maxlen=args.replay_size)
        self.mainDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.start_epoch = self.load_checkpoint()

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

    def save_memory(self):
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

    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            if args.device == 'cpu':
                model_dict = torch.load(self.log + 'model.pt', map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(self.log + 'model.pt')
            self.mainDQN.load_state_dict(model_dict['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), self.log + 'model.pt')

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, self.log + 'model.pt'))


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
        self.load_model()
        i_episode = self.start_epoch
        pushed = False
        printed = False
        updated = False

        while True:
            flag = env.flag()

            # observer
            if flag == 2 : # if in game, observe
                state, got_state = env.state()

            # key press 
            if flag == 2 and got_state : # if in game, action
                action = self.select_action(state) # Select an action
                env.step(action) # step
            elif flag == 10 : # begin screen
                pass # new cloud add here
            else :
                env.release_key() # release all key

            # trajectory push
            if flag == 2 and got_state : # if in game, action
                self.trajectory.push(state, action) # Store the state and action in trajectory
                pushed = False
            elif (flag == 3 or flag == 4) and not pushed : # if win or loss, Store the trajectory in replay memory
                self.push_trajectory(self.trajectory)
                pushed = True

            # printer
            if flag == 2 and got_state : # if in game, action
                print(state)
                printed = False
            elif flag == 3 and not printed : # if win or loss, Store the trajectory in replay memory
                print("rest time")
                printed = True
            elif flag == 4 and not printed : # gameset
                print("gameset")
                printed = True

            # update
            if flag == 4 and not updated : # gameset
                self.load_model()
                self.writer.add_scalar('total_reward', env.point(), i_episode)
                self.save_checkpoint(i_episode)
                i_episode += 1
                updated = True
            else :
                updated = False

if __name__ == "__main__":
    env = Env()
    actor = Actor()
    actor.main()
