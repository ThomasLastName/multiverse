import torch
from torch import nn
from matplotlib import pyplot as plt

import bnns
from bnns.data.bivar_trivial import x_test

BNN = bnns.GPPrior2023BNN(
    nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2)
)

BNN.set_prior_hyperparameters(bw=0.1, scale=1.0, eta=0, gpytorch=False)

try:
    y_prior = BNN.prior_forward(x_test)
except torch._C._LinAlgError:
    try:
        BNN.set_prior_hyperparameters(bw=0.1, scale=1.0, eta=0.00001, gpytorch=False)
        y_prior = BNN.prior_forward(x_test)
        print("")
        print(f"eta={BNN.GP.etas[0]} achieved numerical stability with torch.float")
        print("")
    except torch._C._LinAlgError:
        BNN = BNN.to(torch.double)
        x_test = x_test.to(torch.double)
        y_prior = BNN.prior_forward(x_test)
        print("")
        print(
            f"eta={BNN.GP.etas[0]} achieved numerical stability with torch.double, but not with torch.float"
        )
        print("")


x = x_test[:, 0].squeeze().cpu().numpy()
y = y_prior[0, :, 0].squeeze().cpu().numpy()
plt.plot(x, y)
plt.show()
