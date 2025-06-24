import torch
from torch import nn

torch.manual_seed(2024)

NN = nn.Sequential(
    nn.Linear(5, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 4000),
    nn.ReLU(),
    nn.Linear(4000, 49719),
)
