
import torch
from torch import nn
torch.manual_seed(2024)

NN = nn.Sequential(
        nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),    # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
        nn.Linear(1, 300),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(300, 1)
    )
