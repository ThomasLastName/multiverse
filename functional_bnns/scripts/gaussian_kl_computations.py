
import torch
from tqdm import trange
from bnns.models.bivar_BNN_GP_prior import BNN as self
from quality_of_life.my_base_utils import support_for_progress_bars

n_meas = 10
self.measurement_set = torch.randn(n_meas,2)
self.post_GP_eta = 0.01



### ~~~ 
## ~~~ The ingredients for the original computation
### ~~~

torch.manual_seed(2024) # ~~~ in this implementation, mu_theta is only randomly estimated

mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=False )
Sigma_theta += torch.diag( self.post_GP_eta * torch.ones_like(Sigma_theta.diag()) )
root_Sigma_theta = torch.linalg.cholesky(Sigma_theta)

mu_0, Sigma_0 = self.GP.prior_mu_and_Sigma( self.measurement_set, inv=False, flatten=True, cholesky=False )



### ~~~
## ~~~ A few equivalent ways of computing those ingredients
### ~~~

#
# ~~~ 1. Naive implementation (overflows if n=20)
Sigma_0_inv = torch.linalg.inv(Sigma_0)

with support_for_progress_bars():
    for _ in trange( 10000, desc="Purely naive computation of the Gaussian kl divergence" ):
        kl_naive = (
            (Sigma_0_inv@Sigma_theta).diag().sum()
            - len(mu_0)
            + torch.inner( mu_0-mu_theta, (Sigma_0_inv @ (mu_0-mu_theta)) )
            + Sigma_0.det().log() - Sigma_theta.det().log()
        )/2

#
# ~~~ 2. Compute the determinants using the cholesky decompositions (much faster, but also infinitely more stable)
torch.manual_seed(2024) # ~~~ in this implementation, mu_theta is only randomly estimated

mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=False )
Sigma_theta += torch.diag( self.post_GP_eta * torch.ones_like(Sigma_theta.diag()) )
root_Sigma_theta = torch.linalg.cholesky(Sigma_theta)

mu_0, root_Sigma_0 = self.GP.prior_mu_and_Sigma( self.measurement_set, inv=False, flatten=True, cholesky=True )
Sigma_0_inv = torch.cholesky_inverse(root_Sigma_0)

with support_for_progress_bars():
    for _ in trange( 10000, desc="Computing the determinants using Cholesky decompositions" ):
        kl_using_cholesky_for_the_determinants = (
            (Sigma_0_inv@Sigma_theta).diag().sum()
            - len(mu_0)
            + torch.inner( mu_0-mu_theta, (Sigma_0_inv @ (mu_0-mu_theta)) )
            + 2*root_Sigma_0.diag().log().sum() - 2*root_Sigma_theta.diag().log().sum()
        )/2

assert abs(kl_naive-kl_using_cholesky_for_the_determinants)/abs(kl_naive) < 1e-6

#
# ~~~ 3. Computing the trace as a Euclidean inner product
with support_for_progress_bars():
    for _ in trange( 10000, desc="Computing the trace as a Euclidean inner product" ):
        kl_first = (
            ( Sigma_0_inv * Sigma_theta ).sum()
            - len(mu_0)
            + torch.inner( mu_0-mu_theta, (Sigma_0_inv @ (mu_0-mu_theta)) )
            + 2*root_Sigma_0.diag().log().sum() - 2*root_Sigma_theta.diag().log().sum()
        )/2

assert abs(kl_naive-kl_first)/abs(kl_naive) < 1e-6



### ~~~ 
## ~~~ New and improved computation of the Gaussian KL divergence
### ~~~

torch.manual_seed(2024) # ~~~ in this implementation, mu_theta is only randomly estimated

mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=False )
mu_theta = mu_theta.reshape(-1,2)   # ~~~ unflatten it
Sigma_theta += torch.diag( self.post_GP_eta * torch.ones_like(Sigma_theta.diag()) ) # ~~~ add "stabilizing noise" so that the cholesky decomposition works
root_Sigma_theta = torch.linalg.cholesky(Sigma_theta)

mu_0, root_Sigma_0 = self.GP.prior_mu_and_Sigma( self.measurement_set, inv=False, flatten=False, cholesky=True )
blocks_of_Sigma_0_inv = torch.stack([ torch.cholesky_inverse(K) for K in root_Sigma_0 ])

n,d = mu_0.shape
assert d==self.out_features and n==n_meas
blocks_of_Sigma_theta = torch.stack([ Sigma_theta[j*n:(j+1)*n, j*n:(j+1)*n] for j in range(d)])
assert blocks_of_Sigma_theta.shape == blocks_of_Sigma_0_inv.shape
diff_of_means = (mu_0-mu_theta).reshape(d,n,1)

with support_for_progress_bars():
    for _ in trange( 10000, desc="Using (theoretically efficinet) block diagonal computations" ):
        kl_second = (
            ( Sigma_0_inv * torch.block_diag(*blocks_of_Sigma_theta) ).sum()
            - n
            + (diff_of_means*torch.bmm( blocks_of_Sigma_0_inv, diff_of_means )).sum()
            + 2*root_Sigma_0.diagonal(dim1=1, dim2=2).log().sum() - 2*root_Sigma_theta.diag().log().sum()
        )/2

print("")
print("The first three methods tested all agree to within 6 significant digits.")   # ~~~ not 6 decimmal places
print(f"Wherease the correct value is {kl_naive}, the last method tested gives {kl_second}")
print("")

