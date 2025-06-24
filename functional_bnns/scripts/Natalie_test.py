### ~~~
## ~~~ Config
### ~~~

track_pars = True
make_gif = False
epochs = 30000


### ~~~
## ~~~ Import block
### ~~~

import torch
from torch import nn
from tqdm import trange

import bnns
from bnns.utils import flatten_parameters
from bnns.data.univar_missing_middle_normalized_12 import (
    x_train,
    y_train,
    x_test,
    y_test,
)

from matplotlib import pyplot as plt
from quality_of_life.my_plt_utils import GifMaker
from quality_of_life.my_numpy_utils import moving_average


### ~~~
## ~~~ Define the model
### ~~~

torch.manual_seed(2025)
bnn = bnns.GaussianBNN(
    nn.Unflatten(dim=-1, unflattened_size=(-1, 1)),
    nn.Linear(1, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 1),
    likelihood_std=0.0003,
    prior_type="Xavier",
)
bnn.set_default_uncertainty(type="Xavier")  # ~~~ initialization
# bnn.setup_soft_projection(method="Blundell")    # ~~~ soft projection
means = flatten_parameters(bnn.posterior_mean).clone().detach()
stds = flatten_parameters(bnn.posterior_std).clone().detach()


### ~~~
## ~~~ Define plotting routine
### ~~~

fig, ax = plt.subplots(figsize=(12, 6))


def plot(fig, ax):
    with torch.no_grad():
        fig.tight_layout()
        ax.grid()
        ax.scatter(x_train, y_train, color="green")
        ax.plot(x_test, y_test, "--", color="green")
        for p in bnn(x_test, n=30):
            ax.plot(x_test, p, color="blue", alpha=0.2)
        ax.set_ylim([-1, 1])
    return fig, ax


### ~~~
## ~~~ Train
### ~~~

optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
if make_gif:
    gif = GifMaker()
for _ in trange(epochs):
    loss = bnn.weight_kl() - bnn.estimate_expected_log_likelihood(x_train, y_train)
    loss.backward()
    # bnn.apply_chain_rule_for_soft_projection()
    optimizer.step()
    optimizer.zero_grad()
    # bnn.apply_soft_projection()
    bnn.resample_weights()
    if (_ + 1) % 50 == 0:
        if track_pars:
            means = torch.row_stack(
                [means, flatten_parameters(bnn.posterior_mean).clone().detach()]
            )
            stds = torch.row_stack(
                [stds, flatten_parameters(bnn.posterior_std).clone().detach()]
            )
        if make_gif:
            fig, ax = plot(fig, ax)
            gif.capture()


### ~~~
## ~~~ Plot resutls
### ~~~

if make_gif:
    gif.develop(fps=30)
else:
    fig, ax = plot(fig, ax)
    plt.show()

if track_pars:
    n_plot = 100
    n_trainable_pars = int(means.shape[1] / 6) + 1
    which = torch.randperm(n_trainable_pars)[:n_plot]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for j in trange(n_plot):
        _ = ax[0].plot(moving_average(means[:, j], 100), linewidth=0.5)
        _ = ax[1].plot(moving_average(stds[:, j], 100), linewidth=0.5)
    fig.tight_layout()
    ax[0].grid()
    ax[1].grid()
    plt.show()
