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
from pynput.keyboard import Key, Controller
import subprocess

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--simnum', type=int, default=0, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N', help='log directory')
args = parser.parse_args()

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
        self.trajectory = Trajectory(300)
        self.replay_memory = ReplayMemory(1000)
        self.policy_net = DQN().to(self.device)
        self.keyboard = Controller()

    def load_model(self):
        file = self.log + 'model.pt'
        if os.path.isfile(file):
            if self.device == 'cpu':
                model_dict = torch.load(file, map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(file)
            self.policy_net.load_state_dict(model_dict['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), file)

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, file))

    def save_checkpoint(self, idx):
        file = self.log + 'checkpoint{}.pt'.format(self.simnum)
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx}
        torch.save(checkpoint, file)
        print('Actor {}: Checkpoint saved in '.format(self.simnum), file)

    def load_checkpoint(self):
        file = self.log + 'checkpoint{}.pt'.format(self.simnum)
        if os.path.isfile(file):
            checkpoint = torch.load(file)
            self.simnum = checkpoint['simnum']
            print("Actor {}: loaded checkpoint ".format(self.simnum), '(epoch {})'.format(checkpoint['epoch']), file)
            return checkpoint['epoch']
        else:
            print("Actor {}: no checkpoint found at ".format(self.simnum), file)
            return 0

    def save_memory(self):
        file = self.log + 'memory{}.pt'.format(self.simnum)
        if os.path.isfile(file):
            memory = torch.load(file)
            memory.extend(self.replay_memory)
        else:
            memory = self.replay_memory

        torch.save(memory, file)
        print('Actor {}: Memory saved in '.format(self.simnum), file + ' size = {}'.format(len(self.replay_memory)))
        self.replay_memory.clear()

    def normalization(self, state) :
        normal_state = []
        normal_state.append((state[0] - 108) / 76)
        normal_state.append((state[1] - 176) / 68)
        normal_state.append((state[2] - 324) / 76)
        normal_state.append((state[3] - 176) / 68)
        normal_state.append((state[4] - 216) / 216)
        normal_state.append((state[5] - 126) / 126)
        normal_state.append(state[6] / 24)
        normal_state.append(state[7] / 45)
        normal_state.append(state[8] / 24)
        normal_state.append(state[9] / 45)
        normal_state.append(state[10]/ 60)
        normal_state.append(state[11]/ 90)
        normal_state.append(state[12] * 2 - 1)
        normal_state.append(state[13] * 2 - 1)
        normal_state.append(state[14] * 2 - 1)
        normal_state.append(state[15] * 2 - 1)
        return np.array(normal_state)

    def push_to_replay_memory(self, state2, action, state1, reward ) :
        # push training data to replay memory
        torch_reward = torch.tensor([[reward]], device=self.device, dtype=torch.float)
        torch_action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        torch_state_1 = torch.tensor([state1], device=self.device, dtype=torch.float)
        torch_state_2 = torch.tensor([state2], device=self.device, dtype=torch.float)
        self.replay_memory.push(torch_state_2, torch_action, torch_state_1, torch_reward)

    def reward_function(self, trajectory, score) :
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        normal_state_1 = self.normalization(sa_1.state)
        normal_state_2 = self.normalization(sa_2.state)
        discount = 0.8
        reward = score
        while True :
            self.push_to_replay_memory(normal_state_2, sa_2.action, normal_state_1, reward)

            # pop data from trajectory stack
            sa_1 = sa_2
            sa_2 = trajectory.pop()

            if sa_2 != None and abs(reward) > 0.1 :
                reward *= discount
                normal_state_1 = normal_state_2
                normal_state_2 = self.normalization(sa_2.state)
            else :
                break

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

    def create_environment(self) :
        process = subprocess.Popen(["wine","volleyball.exe", "WINEDEBUG=-all"])
        self.auto_start_new_game()
        self.env = Env(process.pid)

    def auto_start_new_game(self) :
        time.sleep(3)                      # wait for game set animation
        # skip game loading animation
        self.keyboard.press(Key.enter)     # press enter
        time.sleep(0.2)                    # wait for enter trigger the game
        self.keyboard.release(Key.enter)   # press enter
        time.sleep(0.5)                    # wait for game loading menu
        # skip game loading menu animation
        self.keyboard.press(Key.enter)     # press enter
        time.sleep(0.2)                    # wait for enter trigger the game
        self.keyboard.release(Key.enter)   # press enter
        time.sleep(0.5)                    # wait for enter gap
        # choose player1 game
        self.keyboard.press(Key.enter)     # press enter
        time.sleep(0.2)                    # wait for enter trigger the game
        self.keyboard.release(Key.enter)   # press enter
        time.sleep(0.5)                    # wait for game loading player1 game

    def main(self):
        self.create_environment()
        test_epoch = self.start_epoch
        pushed = False
        while True:
            # flag        // 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
            try :
                flag = self.env.flag()
            except :
                self.create_environment()

            if flag == 2 :
                state, got_state = self.env.state()
                if got_state :
                    action = self.select_action(state)
                    self.env.step(action)
                    self.trajectory.push(state, action) # Store the state and action in trajectory
                    pushed = False
            elif flag == 3 :
                if not pushed : # if win or loss, Store the trajectory in replay memory
                    self.env.init()
                    self.reward_function(self.trajectory, self.env.score())
                    pushed = True
            elif flag == 4 :
                if not pushed : # if win or loss, Store the trajectory in replay memory
                    self.env.init()
                    self.reward_function(self.trajectory, self.env.score())
                    test_epoch += 1
                    self.writer.add_scalar('total_reward', self.env.final_score(), test_epoch)
                    self.save_memory()
                    self.load_model()
                    self.save_checkpoint(test_epoch)
                    pushed = True
            elif flag == 10 :
                self.auto_start_new_game()

if __name__ == "__main__":
    actor = Actor()
    actor.main()
