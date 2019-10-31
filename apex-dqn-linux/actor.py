import argparse
from env import Env
from model import DQN
from structure import Trajectory, ReplayMemory
import control as c
import time
import math
import random
import numpy as np
import torch
from tensorboardX import SummaryWriter
import os
import gc

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--simnum', type=int, default=0, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N', help='log directory')
args = parser.parse_args()
torch.manual_seed(1)

class Actor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simnum = args.simnum
        self.log = args.log_directory + args.load_model + '/'
        self.writer = SummaryWriter(self.log + str(self.simnum) + '/')
        self.start_epoch = self.load_checkpoint()
        self.steps_done = 0
        self.n_actions = 6
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.trajectory = Trajectory(3000)
        self.replay_memory = ReplayMemory(3000)
        self.policy_net = DQN().to(self.device)

    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            if self.device == 'cpu':
                model_dict = torch.load(self.log + 'model.pt', map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(self.log + 'model.pt')
            self.policy_net.load_state_dict(model_dict['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), self.log + 'model.pt')

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, self.log + 'model.pt'))

    def save_checkpoint(self, idx):
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx}
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
            return 0

    def save_memory(self):
        if os.path.isfile(self.log + 'memory.pt'):
            try:
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory.extend(self.replay_memory)
            except:
                time.sleep(10)
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory.extend(self.replay_memory)
        else:
            memory = self.replay_memory

        torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
        self.replay_memory.clear()
        print('Actor {}: Memory saved in '.format(self.simnum), self.log + 'memory{}.pt'.format(self.simnum))

    def reward_function(self, trajectory, score) :
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
        test_epoch = self.start_epoch
        pushed = False
        while True:
            # flag        // 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
            flag = env.flag()

            if flag == 2 :
                state, got_state = env.state()
                if got_state :
                    action = self.select_action(state)
                    env.step(action)
                    self.trajectory.push(state, action) # Store the state and action in trajectory
                    print(state)
                    pushed = False
            elif flag == 3 :
                if not pushed : # if win or loss, Store the trajectory in replay memory
                    env.release_key()
                    self.reward_function(self.trajectory, env.score())
                    gc.collect()
                    print("rest time")
                    pushed = True
            elif flag == 4 :
                if not pushed : # if win or loss, Store the trajectory in replay memory
                    env.release_key()
                    self.reward_function(self.trajectory, env.score())
                    self.save_memory()
                    self.load_model()
                    test_epoch += 1
                    self.writer.add_scalar('total_reward', env.final_score(), test_epoch)
                    self.save_checkpoint(test_epoch)
                    gc.collect()
                    print("gameset")
                    pushed = True
            elif flag == 10 :
                pass                          # new cloud add here

if __name__ == "__main__":
    env = Env()
    actor = Actor()
    actor.main()
