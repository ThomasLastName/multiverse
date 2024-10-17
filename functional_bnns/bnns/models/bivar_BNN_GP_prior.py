
import torch
from torch import nn
from bnns.models.bivar_BNN import BNN
from bnns.GPR import simple_mean_zero_RPF_kernel_GP

BNN.GP = simple_mean_zero_RPF_kernel_GP(
        out_features = BNN.out_features,
        eta = 0.19
    )

# from bnns.SequentialGaussianBNN import kernel_matrix
# x = torch.randn(100,2)
# self = BNN
# j = 0
# MU, THETA = self.mu_and_root_Sigma_of_GP_prior( x, inv=True, flatten=True )
# K = self.kernel_scale[j]*kernel_matrix( x, x, self.kernel_bandwidth[j] ) + self.kernel_eta[j]*torch.eye(x.shape[0])
