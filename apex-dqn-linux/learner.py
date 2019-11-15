import control as c
from model import DQN
from structure import ReplayMemory, Transition
import argparse
import datetime
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import subprocess
from collections import deque

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate (default: 1e-4)')
parser.add_argument('--actor-num', type=int, default=10, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',help='load previous model')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N',help='log directory')
args = parser.parse_args()

class Learner():
    def __init__(self):
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.lr = args.lr
        self.alpha = 0.7
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.beta_increment = 1e-6
        self.e = 1e-6
        self.log_interval = 100
        self.update_cycle = 1000
        self.actor_num = args.actor_num
        self.state_size = 16
        self.action_size = 6
        self.load_model()
        self.replay_memory = ReplayMemory(30000)
        self.priority = deque(maxlen=30000)
        self.miss_memory_count = 0
        self.writer = SummaryWriter(self.log)
        self.kill_process_time_stamp = time.time()

    def update_model(self) :
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, train_epoch):
        file = self.log + 'model.pt'
        model_dict = {'state_dict': self.policy_net.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'train_epoch': train_epoch}
        torch.save(model_dict, file)
        print('Learner: Model saved in ', file)

    def load_model(self):
        if args.load_model != '000000000000':
            self.log = args.log_directory + args.load_model + '/'
        else :
            self.log = args.log_directory + datetime.datetime.now().strftime("%y%m%d%H%M%S") + '/'
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.policy_net.train()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.start_epoch = 0
        file = self.log + 'model.pt'
        if os.path.isfile(file):
            model_dict = torch.load(file)
            self.policy_net.load_state_dict(model_dict['state_dict'])
            self.optimizer.load_state_dict(model_dict['optimizer_dict'])
            self.start_epoch = model_dict['train_epoch']
            self.update_model()
            print("Learner: Model loaded from {}(epoch:{})".format(file, str(self.start_epoch)))

    def load_memory(self, simnum):
        file = self.log + 'memory{}.pt'.format(simnum)
        if os.path.isfile(file):
            try :
                memory = torch.load(file)
                os.remove(file)
                self.replay_memory.extend(memory['replay_memory'])
                self.priority.extend(memory['priority'])
                print('Memory loaded from ', file, 'size = {}'.format(len(self.replay_memory)))
            except :
                print("Memory loaded premission denied", file)
        else :
            print('Memory missing', file)
            self.miss_memory_count += 1

    def sample(self, train_epoch):
        priority = (np.array(self.priority) + self.e) ** self.alpha
        weight = (len(priority) * priority) ** -self.beta
        weight /= weight.max()
        self.weight = torch.tensor(weight, dtype=torch.float)
        priority = torch.tensor(priority, dtype=torch.float)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(priority, self.BATCH_SIZE, replacement=True)
        return sampler

    def optimize_model(self, train_epoch):
        if len(self.replay_memory) < self.BATCH_SIZE*100:
            return False, 0

        self.optimizer.zero_grad()
        self.policy_net.train()
        self.target_net.eval()
        x_stack = torch.zeros(0, self.state_size).to(self.device)
        y_stack = torch.zeros(0, self.action_size).to(self.device)
        w = []
        self.beta = min(1, self.beta_init + train_epoch * self.beta_increment)

        for idx in self.sample(train_epoch):
            state, action, next_state, reward = self.replay_memory.memory[idx]
            state_value = self.policy_net(state)
            next_state_value = self.policy_net(next_state)
            tderror = reward + self.GAMMA * self.target_net(next_state)[0, torch.argmax(next_state_value, 1)] - state_value[0, action]
            state_value[0, action] += tderror
            x_stack = torch.cat([x_stack, state.data], 0)
            y_stack = torch.cat([y_stack, state_value.data], 0)
            w.append(self.weight[idx])
            self.priority[idx] = tderror.abs().item()
        pred = self.policy_net(x_stack)
        w = torch.tensor(w, dtype=torch.float, device=self.device)
        loss = torch.dot(F.smooth_l1_loss(pred, y_stack.detach(), reduction='none').sum(1), w.detach())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return True, loss / self.BATCH_SIZE

    def main(self):
        train_epoch = self.start_epoch
        self.save_model(train_epoch)

        while True:
            is_optimized, loss = self.optimize_model(train_epoch)
            if is_optimized :
                train_epoch += 1
                self.writer.add_scalar('loss', loss.item(), train_epoch)

                if train_epoch % self.log_interval == 0:
                    print('Train Epoch: {} \tLoss: {}'.format(train_epoch, loss.item()))
                    self.writer.add_scalar('replay size', len(self.replay_memory), train_epoch)
                    if (train_epoch // self.log_interval) % self.actor_num == 0:
                        self.save_model(train_epoch)
                    self.load_memory((train_epoch // self.log_interval) % self.actor_num)
                    subprocess.run(["killall", "-9", "winedbg"])

                if train_epoch % self.update_cycle == 0:
                    self.update_model()

                if self.miss_memory_count > 5 :
                    subprocess.run(["killall", "-9", "volleyball.exe"])
                    self.miss_memory_count = 0
            else :
                print("Memory not enough")
                for i in range(self.actor_num):
                    if os.path.isfile(self.log + '/memory{}.pt'.format(i)) :
                        self.load_memory(i)
                    time.sleep(1)

if __name__ == "__main__":
    learner = Learner()
    learner.main()

