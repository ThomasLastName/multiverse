
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import os
import torch
from torch import nn

#
# ~~~ The guts of the model
from bnns.SequentialGaussianBNN import SequentialGaussianBNN



### ~~~
## ~~~ Config
### ~~~

#
# ~~~ Misc.
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"



### ~~~
## ~~~ Define the network architecture
### ~~~

from bnns.models.univar_BNN import BNN
from bnns.models.univar_NN  import  NN
NN, BNN = NN.to(DEVICE), BNN.to(DEVICE)



### ~~~
## ~~~ Define the data
### ~~~

from bnns.data.univar_missing_middle import x_train, y_train, x_test, y_test, ground_truth
x_train, y_train, x_test, y_test = x_train.to(DEVICE), y_train.to(DEVICE), x_test.to(DEVICE), y_test.to(DEVICE)

BNN.prior_forward(x_test)   