
import torch
from torch import nn
from bnns.models.bivar_BNN import BNN

BNN.use_GP_prior = True
BNN.prior_GP_kernel_bandwidth = torch.tensor(BNN.out_features*[0.1])
BNN.prior_GP_kernel_scale = torch.tensor(BNN.out_features*[1.])
BNN.prior_GP_kernel_eta = torch.tensor(BNN.out_features*[0.19])

# from bnns.SequentialGaussianBNN import kernel_matrix
# x = torch.randn(100,2)
# self = BNN
# j = 0
# MU, THETA = self.mu_and_root_Sigma_of_GP_prior( x, inv=True, flatten=True )
# K = self.kernel_scale[j]*kernel_matrix( x, x, self.kernel_bandwidth[j] ) + self.kernel_eta[j]*torch.eye(x.shape[0])
