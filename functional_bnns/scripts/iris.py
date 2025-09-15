
import numpy as np
from tqdm import trange
from sklearn import datasets


import torch
import torch.nn as nn
import torch.optim as optim

from bnns import GaussianBNN
torch.manual_seed(2025)

#
# ~~~ Load iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target 
x, y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
x.shape, y.shape

#
# ~~~ Define Model
bnn = GaussianBNN(
    nn.Linear(4, 100),
    nn.ReLU(),
    nn.Linear(100, 3)
)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(bnn.parameters(), lr=0.01)

#
# ~~~ Train model
kl_weight = 0.0001
for step in trange(3000):
    bnn.resample_weights()
    pre = bnn(x)
    ce = ce_loss(pre,y)
    kl = bnn.weight_kl()
    cost = ce + kl_weight*kl    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


_, predicted = torch.max(pre.data, 1)
n_wrong = (bnn(x,30).argmax(dim=-1).mode(dim=0).values == y).sum()
correct = 1 - n_wrong/len(y)
print('- Accuracy: %f %%' % (100 * correct))
print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))