import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        input_size = 16
        hidden_size = 128
        output_size = 6
        
        super(DQN, self).__init__()
        self.vfc1 = nn.Linear(input_size, hidden_size)
        self.vfc2 = nn.Linear(hidden_size, 1)
        self.afc1 = nn.Linear(input_size, hidden_size)
        self.afc2 = nn.Linear(hidden_size, output_size)
        torch.nn.init.normal_(self.vfc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.vfc2.weight, 0, 0.02)
        torch.nn.init.normal_(self.afc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.afc2.weight, 0, 0.02)

    def forward(self, x):
        a = F.relu(self.afc1(x))
        a = self.afc2(a)
        av = torch.mean(a, 1, True)
        av = av.expand_as(a)
        v = F.relu(self.vfc1(x))
        v = self.vfc2(v)
        v = v.expand_as(a)
        x = a - av + v
        return x