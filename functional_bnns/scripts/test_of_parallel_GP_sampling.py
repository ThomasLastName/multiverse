
import torch
from bnns.models.slosh_70_15_15_centered_pca_BNN_GP_prior import BNN, x_test

x = x_test
self = BNN
mu, root_Sigma = self.mu_and_root_Sigma_of_GP_prior(x)
assert mu.shape==(700,9)
assert root_Sigma.shape==(9,700,700)

torch.manual_seed(2024)
Z = torch.randn( size=( 150, x.shape[0] ), device=x.device, dtype=x.dtype )
not_parallel = torch.row_stack([ (mu+(root_Sigma@Z[j]).T).flatten() for j in range(150) ])
assert not_parallel.shape == ( 150, x.shape[0]*self.out_features )

torch.manual_seed(2024)
Z = torch.randn( size=( 150, x.shape[0] ), device=x.device, dtype=x.dtype )
Sz = root_Sigma@Z.T
parallel = torch.row_stack([ (mu + Sz[:,:,j].T).flatten() for j in range(150) ])
assert parallel.shape == ( 150, x.shape[0]*self.out_features )

assert torch.allclose(parallel,not_parallel)


