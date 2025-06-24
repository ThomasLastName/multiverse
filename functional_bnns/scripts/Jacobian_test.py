### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from torch import nn

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset


### ~~~
## ~~~ Define the network architecture
### ~~~

NN = nn.Sequential(
    nn.Unflatten(
        dim=-1, unflattened_size=(-1, 1)
    ),  # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
)


### ~~~
## ~~~ Make up some data
### ~~~

#
# ~~~ Synthetic (noisy) training data
torch.manual_seed(2024)
x_train = 2 * torch.rand(size=(30,)) ** 2 - 1  # ~~~ uniformly random points in [-1,1]
x_train = x_train
y_train = torch.cos(x_train) + 0.01 * torch.randn(size=(30,))
y_train = y_train.reshape(-1, 1)


### ~~~
## ~~~ Train a conventional neural network, for reference
### ~~~

optimizer = torch.optim.Adam(NN.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()


#
# ~~~ Compute the gradient using a combination of backprop and the chain rule
SSGE = lambda y: ((y - y_train) / (2 * 30)).reshape(1, -1)

y_pred = NN(x_train)
with torch.no_grad():
    g = SSGE(NN(x_train))

loss = 4 * (g @ y_pred).squeeze()
loss.backward()
for p in NN.parameters():
    pass

print(f"The last partial derivative by chain rule is {p.grad}")
optimizer.zero_grad()


#
# ~~~ Compute the gradient using only backprop
loss = loss_fn(NN(x_train), y_train)
loss.backward()
for p in NN.parameters():
    pass

print(f"The last partial derivative by backprop is {p.grad}")
optimizer.zero_grad()
