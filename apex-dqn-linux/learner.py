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
        print('{} Learner: Model saved in '.format(train_epoch), file)

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

    def optimize_model(self, train_epoch):
        if len(self.replay_memory) < self.BATCH_SIZE :
            return False, 0

        self.beta = min(1, self.beta_init + train_epoch * self.beta_increment)
        priority = (np.array(self.priority) + self.e) ** self.alpha
        weight = (len(priority) * priority) ** -self.beta
        weight /= weight.max()
        priority = torch.tensor(priority, dtype=torch.float)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(priority, self.BATCH_SIZE, replacement=True)

        batch_weight = []
        batch_state = torch.zeros(0, self.state_size).to(self.device)
        batch_action = torch.zeros(0, 1, dtype=torch.long).to(self.device)
        batch_next_state = torch.zeros(0, self.state_size).to(self.device)
        batch_reward = torch.zeros(0, 1).to(self.device)
        batch_idx = []
        for idx in sampler :
            state, action, next_state, reward = self.replay_memory.memory[idx]
            batch_state = torch.cat([batch_state, state.data], 0)
            batch_action = torch.cat([batch_action, action.data], 0)
            batch_next_state = torch.cat([batch_next_state, next_state.data], 0)
            batch_reward = torch.cat([batch_reward, reward.data], 0)
            batch_weight.append(weight[idx])
            batch_idx.append(idx)
        batch_weight = torch.tensor(batch_weight, dtype=torch.float, device=self.device)

        state_action_value = self.policy_net(batch_state).gather(1, batch_action)
        next_state_action_value = self.target_net(batch_next_state).gather(1, torch.argmax(self.policy_net(batch_next_state), 1).unsqueeze(1))
        tderror = batch_reward + self.GAMMA * next_state_action_value - state_action_value
        except_state_action_value = state_action_value + tderror
        loss = F.smooth_l1_loss(state_action_value, except_state_action_value.detach(), reduction='none').squeeze(1)
        loss = torch.dot(loss, batch_weight.detach())

        count = 0
        for idx in batch_idx :
            self.priority[idx] = tderror[count].abs().item()
            count += 1

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
                    print("kill all process")
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

