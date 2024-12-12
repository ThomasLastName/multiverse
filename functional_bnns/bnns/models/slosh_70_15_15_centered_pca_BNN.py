
import torch
from torch import nn
from bnns.SequentialGaussianBNN import SequentialGaussianBNN
from bnns.data.slosh_70_15_15_centered_pca import r  # ~~~ the number of principal components: the output dimension of the NN
torch.manual_seed(2024)

BNN = SequentialGaussianBNN(
        nn.Linear(5, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, r)
    )