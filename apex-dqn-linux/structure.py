import random
from collections import namedtuple

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

    def extend(self, replay_memory) :
        for i in range(len(replay_memory)) :
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = replay_memory.memory[i]
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self) :
        del self.memory
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)