import torch
from torch import nn
from torch.func import jacrev, functional_call
from tqdm import trange


### ~~~
## ~~~ Config.
### ~~~

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2024)


### ~~~
## ~~~ Make up a trained neural network
### ~~~

#
# ~~~ Start with an untrained network
NN = nn.Sequential(
    nn.Unflatten(
        dim=-1, unflattened_size=(-1, 1)
    ),  # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
).to(DEVICE)

#
# ~~~ Make up some data
n_train = 50
n_test = 500
noise = 0.2
f = (
    lambda x: 2 * torch.cos(torch.pi * (x + 0.2))
    + torch.exp(2.5 * (x + 0.2)) / 2.5
    - 2.25
)

#
# ~~~ Synthetic (noisy) training data
x_train = (
    2 * torch.rand(size=(n_train,), device=DEVICE) ** 2 - 1
)  # ~~~ uniformly random points in [-1,1]
x_train = x_train.sign() * x_train.abs() ** (1 / 6)
y_train = f(x_train) + noise * torch.randn(size=(n_train,), device=DEVICE)

#
# ~~~ Synthetic (noise-less) test data
x_test = torch.linspace(-1.2, 1.2, n_test, device=DEVICE)
y_test = f(x_test)

#
# ~~~ Reshape y data in order to be consistent with the shape returned by a model with final layer nn.Linear(m,1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#
# ~~~ Train
optimizer = torch.optim.Adam(NN.parameters(), lr=0.00005)
for j in trange(2000):
    loss = ((NN(x_test) - y_test) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


### ~~~
## ~~~ Compute the Jacobian of the trained network
### ~~~


#
# ~~~ Compute by hand the Jacobian with respect to only the last layer of parameters
def compute_by_hand(NN, x):
    J_beta = x
    for j in range(len(NN) - 1):
        J_beta = NN[j](J_beta)
    return J_beta


J_beta = compute_by_hand(NN, x_train)
rank = torch.linalg.matrix_rank(J_beta).item()
assert rank == 14

#
# ~~~ Compute the full Jacobian using torch.func (second example at https://pytorch.org/docs/stable/func.migrating.html#functorch-make-functional)
jacobians = jacrev(functional_call, argnums=1)(
    NN, dict(NN.named_parameters()), (x_train,)
)  # ~~~ a dictionary with the same keys as NN.named_parameters()
J_beta_func = jacobians["5.weight"].squeeze()

#
# ~~~ Observe that they are the same
assert torch.allclose(J_beta, J_beta_func)
