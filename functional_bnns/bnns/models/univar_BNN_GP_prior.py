

import torch
from torch import nn
from bnns.models.univar_BNN import BNN
from bnns.GPPriorBNN import simple_mean_zero_RPF_kernel_GP as GP

BNN.GP = GP( out_features=BNN.out_features, eta=0.001 )

