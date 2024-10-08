
import torch
from bnns.data.slosh_70_15_15 import x_train

interpolary_grid = x_train + 0.05*torch.randn_like(x_train)
dists = torch.cdist(interpolary_grid,x_train).min(dim=1).values
