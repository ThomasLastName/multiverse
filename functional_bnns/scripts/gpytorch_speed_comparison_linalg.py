
import torch
from bnns.GPR import simple_mean_zero_RPF_kernel_GP as GPR
from bnns.GPR import GPYBackend
from copy import copy
from tqdm import trange
from quality_of_life.my_base_utils import support_for_progress_bars

#
# ~~~ Make up fake data
n_samples = 2000
n_features = 50
device = "cuda"
x_test = torch.randn( n_samples, n_features, device=device )

#
# ~~~ Define models
gpr = GPR( out_features=n_features )
_,_ = gpr.prior_mu_and_Sigma( x_test )
backend = GPYBackend( x_test, out_features=gpr.out_features, bandwidths=copy(gpr.bandwidths), scales=gpr.scales, etas=gpr.etas )

with support_for_progress_bars():
    for C in (True,False):
        for A in (True,False):
            if not (C and not A):
                for _ in trange(20): mu_prior, Sigma_prior         = gpr.prior_mu_and_Sigma( x_test, add_stabilizing_noise=A, cholesky=C )
                for _ in trange(20): mu_prior_gpy, Sigma_prior_gpy = gpr.prior_mu_and_Sigma( x_test, add_stabilizing_noise=A, gpytorch=True, cholesky=C )
                for _ in trange(20): mu_prior, Sigma_prior         = backend.prior_mu_and_Sigma( x_test, add_stabilizing_noise=A, cholesky=C )
                print("")
                print(f"Results are effectively the same. Max absolute error {( Sigma_prior - Sigma_prior_gpy ).abs().max().item()}")
                print("")

