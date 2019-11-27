import argparse
from env import Env
from model import DQN
from structure import Trajectory, ReplayMemory, Priority
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
from collections import deque

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
        self.start_epoch, self.base_reward = self.load_checkpoint()
        self.n_actions = 6
        self.discount = 0.98
        self.trajectory = Trajectory(1000)
        self.replay_memory = ReplayMemory(10000)
        self.priority = Priority(10000)
        self.policy_net = DQN().to(self.device)
        self.keyboard = Controller()

    def save_model(self, final_score):
        file = self.log + 'best_model{}.pt'.format(final_score)
        torch.save(self.policy_net.state_dict(), file)
        print('Actor: Model saved in ', file)

    def load_model(self):
        file = self.log + 'model.pt'
        if os.path.isfile(file):
            if self.device == 'cpu':
                model_dict = torch.load(file, map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(file)

            try :
                self.policy_net.load_state_dict(model_dict['state_dict'])
            except :
                self.policy_net.load_state_dict(model_dict)

            print('Actor {}: Model loaded from '.format(self.simnum), file)

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, file))

    def save_checkpoint(self, idx):
        file = self.log + 'checkpoint{}.pt'.format(self.simnum)
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx,
                      'base_reward': self.base_reward}
        torch.save(checkpoint, file)
        print('Actor {}: Checkpoint saved in '.format(self.simnum), file)

    def load_checkpoint(self):
        file = self.log + 'checkpoint{}.pt'.format(self.simnum)
        if os.path.isfile(file):
            checkpoint = torch.load(file)
            self.simnum = checkpoint['simnum']
            print("Actor {}: loaded checkpoint ".format(self.simnum), '(epoch {})'.format(checkpoint['epoch']), file)
            return checkpoint['epoch'], checkpoint['base_reward']
        else:
            print("Actor {}: no checkpoint found at ".format(self.simnum), file)
            return 0, 1

    def save_memory(self):
        file = self.log + 'memory{}.pt'.format(self.simnum)
        if os.path.isfile(file):
            memory = torch.load(file)
            memory['replay_memory'].extend(self.replay_memory)
            memory['priority'].extend(self.priority)
        else:
            memory = {'replay_memory': self.replay_memory, 'priority': self.priority}

        torch.save(memory, file)
        print('Actor {}: Memory saved in '.format(self.simnum), file + ' size = {}'.format(len(self.replay_memory)))
        self.replay_memory.clear()
        self.priority.clear()

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

    def reward_function(self, trajectory, priority, score) :
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        reward = score - self.base_reward
        while True :
            self.push_to_replay_memory(sa_2.state, sa_2.action, sa_1.state, reward + self.base_reward)

            # pop data from trajectory stack
            sa_1 = sa_2
            sa_2 = trajectory.pop()

            if sa_2 != None :
                reward *= self.discount
            else :
                break

        self.base_reward *= self.discount
        self.priority.extend_list(priority[1:len(priority)])
        self.trajectory.clear()

    def select_action(self, state):
        self.policy_net.eval()
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        sample = random.random()
        Q = self.policy_net(state)
        maxv, action = torch.max(Q, 1)

        if sample > 0.9:
            action = random.randrange(self.n_actions)
        else:
            action = action.item()

        return maxv.item(), action

    def create_environment(self) :
        process = subprocess.Popen(["wine","volleyball.exe"])
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
        self.load_model()
        self.create_environment()
        test_epoch = self.start_epoch
        pushed = False
        estimate = 0
        priority_list = []
        while True:
            try :
                # flag        // 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
                flag = self.env.flag()

                if flag == 2 :
                    state, got_state = self.env.state()
                    if got_state :
                        state = self.normalization(state)
                        maxv, action = self.select_action(state)
                        self.env.step(action)
                        priority_list.append(abs(self.discount * maxv - estimate))
                        estimate = maxv
                        self.trajectory.push(state, action) # Store the state and action in trajectory
                        pushed = False
                elif flag == 3 :
                    if not pushed : # if win or loss, Store the trajectory in replay memory
                        self.env.init()
                        estimate = 0
                        self.reward_function(self.trajectory, priority_list, self.env.score())
                        priority_list = []
                        pushed = True
                elif flag == 4 :
                    if not pushed : # if win or loss, Store the trajectory in replay memory
                        self.env.init()
                        self.reward_function(self.trajectory, priority_list, self.env.score())
                        priority_list = []
                        test_epoch += 1
                        final_score = self.env.final_score()
                        self.writer.add_scalar('total_reward', final_score, test_epoch)
                        self.save_model(final_score)
                        self.save_memory()
                        self.load_model()
                        self.save_checkpoint(test_epoch)
                        pushed = True
                elif flag == 10 :
                    self.auto_start_new_game()

            except :
                self.env.init()
                self.trajectory.clear()
                self.create_environment()
                pushed = False

if __name__ == "__main__":
    actor = Actor()
    actor.main()
