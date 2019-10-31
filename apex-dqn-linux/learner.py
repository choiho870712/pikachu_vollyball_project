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
import gc

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--actor-num', type=int, default=10, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',help='load previous model')
parser.add_argument('--log-directory', type=str, default='log/', metavar='N',help='log directory')
args = parser.parse_args()
torch.manual_seed(1)

class Learner():
    def __init__(self):
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.log_interval = 100
        self.update_cycle = 1000
        self.actor_num = args.actor_num
        self.load_model()
        self.replay_memory = ReplayMemory(30000)
        self.writer = SummaryWriter(self.log)

    def update_model(self) :
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, train_epoch):
        model_dict = {'state_dict': self.policy_net.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'train_epoch': train_epoch}
        torch.save(model_dict, self.log + 'model.pt')
        print('Learner: Model saved in ', self.log + 'model.pt')

    def load_model(self):
        if args.load_model != '000000000000':
            self.log = args.log_directory + args.load_model + '/'
        else :
            self.log = args.log_directory + datetime.datetime.now().strftime("%y%m%d%H%M%S") + '/'
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.start_epoch = 0
        if os.path.isfile(self.log + 'model.pt'):
            model_dict = torch.load(self.log + 'model.pt')
            self.policy_net.load_state_dict(model_dict['state_dict'])
            self.optimizer.load_state_dict(model_dict['optimizer_dict'])
            self.start_epoch = model_dict['train_epoch']
            self.update_model()
            print("Learner: Model loaded from {}(epoch:{})".format(self.log + 'model.pt', str(self.start_epoch)))

    def load_memory(self, simnum):
        if os.path.isfile(self.log + 'memory{}.pt'.format(simnum)):
            try:
                memory = torch.load(self.log + 'memory{}.pt'.format(simnum))
                self.replay_memory.extend(memory)
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                memory.clear()
                torch.save(memory, self.log + 'memory{}.pt'.format(simnum))
            except:
                time.sleep(10)
                memory = torch.load(self.log + 'memory{}.pt'.format(simnum))
                self.replay_memory.extend(memory)
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                memory.clear()
                torch.save(memory, self.log + 'memory{}.pt'.format(simnum))
        else:
            print("=> Learner: no memory found at ", self.log + 'memory{}.pt'.format(simnum))

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return False, 0
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

        return True, loss

    def main(self):
        train_epoch = self.start_epoch
        self.save_model(train_epoch)

        while True:
            is_optimized, loss = self.optimize_model()
            if is_optimized :
                train_epoch += 1
                self.writer.add_scalar('loss', loss.item(), train_epoch)
                gc.collect()
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
