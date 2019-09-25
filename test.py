import cv2
from mss import mss
import numpy as np
import math
import random
import time
import pynput
from pynput.keyboard import Key, Controller
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Env(object) :

    def __init__(self) :
        self.bbox = {'top': 50, 'left': 0, 'width': 430, 'height': 280}
        self.sct = mss()
        self.keyboard = Controller()
        self.lower_red = np.array([0, 200, 120])
        self.upper_red = np.array([10, 255, 150])
        self.lower_yellow = np.array([20, 120, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.img_high = 100
        self.img_width = 100
        self.gaming_status = "gameset"
        self.time_stamp = 0

    def reset(self):
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.down)
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
        self.keyboard.release(Key.enter)

    def step(self, num):
        if num == 0: # up
            self.keyboard.release(Key.down)
            self.keyboard.press(Key.up)
        elif num == 1: # down
            self.keyboard.release(Key.up)
            self.keyboard.press(Key.down)
        elif num == 2: # left
            self.keyboard.release(Key.right)
            self.keyboard.press(Key.left)
        elif num == 3: # right
            self.keyboard.release(Key.left)
            self.keyboard.press(Key.right)
        elif num == 4: # enter
            self.keyboard.press(Key.enter)
        elif num == 5: # release all
            self.keyboard.release(Key.up)
            self.keyboard.release(Key.down)
            self.keyboard.release(Key.left)
            self.keyboard.release(Key.right)
            self.keyboard.release(Key.enter)

    def get_screen(self) :

        def preprocessing(img) :
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_red = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
            img_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
            return img_red + img_yellow

        def get_reward(img) :
            left_boom = np.sum(img[-1:, int(self.img_width/2):])
            right_boom = np.sum(img[-1:, :int(self.img_width/2)])
            reward = 0
            if self.gaming_status == "gaming" :
                if left_boom > right_boom :
                    reward = -1
                elif left_boom < right_boom :
                    reward = 1

            return reward

        def set_status(img, reward) :

            def compare_area_ratio(img, ratio) :
                if np.sum(img) >= np.size(img)*255*ratio :
                    return True
                else :
                    return False

            if self.gaming_status == "gaming" :
                if reward != 0 :
                    self.gaming_status = "next_stage"
                    self.time_stamp = time.time()
                    print(self.gaming_status)
            elif self.gaming_status == "gameset" :
                if compare_area_ratio(img, 0.2) :           # wait for start( find start page )
                    self.gaming_status = "waiting_for_start"
                    print(self.gaming_status)
            elif self.gaming_status == "waiting_for_start" :
                if np.sum(img) == 0 :                       # black screen
                    self.gaming_status = "start"
                    self.time_stamp = time.time()
                    print(self.gaming_status)
            elif self.gaming_status == "start" :
                if time.time() - self.time_stamp > 2.872 :   # start delay end
                    self.gaming_status = "gaming"
                    print(self.gaming_status)
            elif self.gaming_status == "next_stage" : 
                if time.time() - self.time_stamp > 2.372 :    # next delay end
                    self.gaming_status = "gaming"                         # restart( find gameset flag )
                    print(self.gaming_status)
                elif compare_area_ratio(img[ int(self.img_high*0.05):int(self.img_high*0.3) , : ], 0.3) :
                    self.gaming_status = "gameset" 
                    print(self.gaming_status)

        img = cv2.resize(np.array(self.sct.grab(self.bbox)), (self.img_high, self.img_width))
        mask = preprocessing(img)
        reward = get_reward(mask)
        set_status(mask, reward)

        # transform state
        #mask = mask[:int(self.img_high*0.9), :]
        state = np.expand_dims(mask, axis=0)  # shape(H, W) to shape(C, H, W)
        state = np.expand_dims(state, axis=0)  # shape(C, H, W)  to shape(B, C, H, W)
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.tensor(state, device=device, dtype=torch.float)
            
        return state, reward, self.gaming_status

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

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Actor(object) :

    def __init__(self, screen_height, screen_width, n_actions, device) :
        self.steps_done = 0
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.n_actions = n_actions
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.policy_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.replay_memory = ReplayMemory(10000)

    def update_model(self) :
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_trajectory(self, trajectory, reward) :
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        discount = 0.99
        while ( sa_1 != None and sa_2 != None ) :
            torch_reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            self.replay_memory.push(sa_2.state, sa_2.action, sa_1.state, torch_reward)
            reward = reward * discount
            if reward > -0.1 and reward < 0 :
                reward = 0.1
                discount = 1.01
            elif reward > 1 :
                discount = 1
            sa_1 = sa_2
            sa_2 = trajectory.pop()

        trajectory.clear()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 0.01
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
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



env = Env()
env.reset() # release all key
state, reward, gaming_status = env.get_screen()

actor = Actor(np.shape(state)[2], np.shape(state)[3], 6, device)
trajectory = Trajectory(1000)

keyboard = Controller()

i_episode = 0
TARGET_UPDATE = 2
while "Screen capturing":
    # get screen
    state, reward, gaming_status = env.get_screen()

    # show state
    cv2.imshow("show", cv2.resize(state.cpu().squeeze(0).squeeze(0).numpy(), (200, 200)))
    cv2.waitKey(50)

    # if in game, action
    if gaming_status == "gaming" :
        isPressed = False

        # Select and perform an action
        action = actor.select_action(state)
        env.step(action)

        # Store the state and action in trajectory
        trajectory.push(state, action)

    # if win or loss, Store the trajectory in replay memory
    elif reward != 0 :
        actor.push_trajectory(trajectory, reward)
        i_episode += 1

    # optimize model when rest time
    elif gaming_status == "gameset" :
        env.reset() # release all key
        actor.optimize_model()

    elif gaming_status == "waiting_for_start" :
        keyboard.press(Key.enter)
        time.sleep(0.5)
        keyboard.release(Key.enter)

    elif gaming_status == "next_stage" :
        env.reset() # release all key

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        actor.update_model()