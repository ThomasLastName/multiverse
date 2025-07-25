import torch
from torch import nn

torch.manual_seed(2024)

NN = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(100, 2),
)
