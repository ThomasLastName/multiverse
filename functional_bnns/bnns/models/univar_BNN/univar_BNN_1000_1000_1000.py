
import torch
from torch import nn
from bnns.SequentialGaussianBNN import SequentialGaussianBNN
torch.manual_seed(2024)

BNN = SequentialGaussianBNN(
        nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),    # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
        nn.Linear(1, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1)
    )

def generate_meas_set(self=BNN):
    eps = .1
    random_interpolary_grid = torch.rand(80,) * 2*(1+eps) - (1+eps)
    self.measurement_set = torch.cat([
            random_interpolary_grid.to(
                    device = self.last_seen_x.device,
                    dtype  =  self.last_seen_x.dtype
                ),
            self.last_seen_x
        ]) if hasattr(self,"last_seen_x") else random_interpolary_grid

BNN.sample_new_measurement_set = generate_meas_set
