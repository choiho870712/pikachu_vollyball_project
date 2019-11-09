import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        input_size = 16
        hidden_size = 100
        output_size = 6
        
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer.weight.data.normal_(0, 0.1)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)