import torch
from torch import nn

torch.manual_seed(2024)

NN = nn.Sequential(
    nn.Linear(5, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 49719)
)
