import torch
from torch import nn
from tqdm import trange
import matplotlib.pyplot as plt

#
# ~~~ Make up data
torch.manual_seed(2025)
f_x = lambda x: x**2/64 - 5*x/4 + 25
x_train = torch.linspace(30,50,21).reshape(-1,1)
y_train = f_x(x_train)

#
# ~~~ Standardize data; MODEL FAILS TO TRAIN IF YOU COMMENT OUT THESE 2 LINES
# x_train = ( x_train - x_train.mean() )/x_train.std()
# y_train = ( y_train - y_train.mean() )/y_train.std()

#
# ~~~ Set up model
net = nn.Sequential( nn.Linear(1,900), nn.ReLU(), nn.Linear(900,900), nn.ReLU(), nn.Linear(900,1) )

#
# ~~~ Train
optimizer = torch.optim.Adam(net.parameters(), lr=0.005 )
for _ in trange(10000):
    y_pred = net(x_train)
    loss = torch.sum(torch.abs(y_pred - y_train))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#
# ~~~ Plot the results
plt.scatter( x_train, y_train, color="green", label="Training Data" )
plt.plot( x_train, net(x_train).detach(), label="Fitted Model" )
plt.show()
