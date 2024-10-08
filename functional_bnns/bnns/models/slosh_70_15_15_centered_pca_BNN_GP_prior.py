
import torch
from torch import nn
from bnns.data.slosh_70_15_15_centered_pca import x_test
from bnns.models.slosh_70_15_15_centered_pca_BNN import BNN

BNN.use_GP_prior = True
BNN.prior_GP_kernel_bandwidth = torch.tensor(BNN.out_features*[15.])
BNN.prior_GP_kernel_scale = torch.tensor(BNN.out_features*[1.])
BNN.prior_GP_kernel_eta = torch.tensor(BNN.out_features*[0.19])
