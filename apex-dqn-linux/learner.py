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

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate (default: 1e-4)')
parser.add_argument('--actor-num', type=int, default=10, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',help='load previous model')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N',help='log directory')
args = parser.parse_args()

class Learner():
    def __init__(self):
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.lr = args.lr
        self.log_interval = 100
        self.update_cycle = 1000
        self.actor_num = args.actor_num
        self.load_model()
        self.replay_memory = ReplayMemory(30000)
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
        try :
            memory = torch.load(file)
            self.replay_memory.extend(memory)
            memory.clear()
            os.remove(file)
            print('Memory loaded from ', file, 'size = {}'.format(len(self.replay_memory)))
        except :
            print('Memory missing', file)
            if time.time() - self.kill_process_time_stamp > 300 :
                subprocess.run(["killall", "-9", "volleyball.exe"])
                print("kill all process")
                self.kill_process_time_stamp = time.time()

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return False, 0
        transitions = self.replay_memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # next_state_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        next_state_values = self.target_net(next_states).gather(1, torch.argmax(self.policy_net(next_states), 1).unsqueeze(1)).detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return True, loss/self.BATCH_SIZE

    def main(self):
        train_epoch = self.start_epoch
        self.save_model(train_epoch)

        while True:
            is_optimized, loss = self.optimize_model()
            if is_optimized :
                train_epoch += 1
                self.writer.add_scalar('loss', loss.item(), train_epoch)

                if train_epoch % self.log_interval == 0:
                    print('Train Epoch: {} \tLoss: {}'.format(train_epoch, loss.item()))
                    self.writer.add_scalar('replay size', len(self.replay_memory), train_epoch)
                    if (train_epoch // self.log_interval) % self.actor_num == 0:
                        self.save_model(train_epoch)
                    self.load_memory((train_epoch // self.log_interval) % self.actor_num)

                if train_epoch % self.update_cycle == 0:
                    self.update_model()
            else :
                print("Memory not enough")
                for i in range(self.actor_num):
                    if os.path.isfile(self.log + '/memory{}.pt'.format(i)) :
                        self.load_memory(i)
                    time.sleep(1)

if __name__ == "__main__":
    learner = Learner()
    learner.main()

