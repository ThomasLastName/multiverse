
import torch
from bnns.GPR import RPF_kernel_GP as GPR
from bnns.GPR import GPY
# from bnns.GPR import MultiOutputGP as GPY
from copy import copy
from tqdm import trange
from quality_of_life.my_base_utils import support_for_progress_bars
from matplotlib import pyplot as plt

#
# ~~~ Make up fake data
# torch.set_default_dtype(torch.float64)
n_train = 51
n_test = 2001
n_features = 5
n_pred = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2025)

x_train = torch.stack(n_features*[torch.linspace(-1,1,n_train)]).T
y_train = x_train.abs() + 0.05*torch.randn_like(x_train)

x_test = torch.stack(n_features*[torch.linspace(-1,1,n_test)]).T
y_test = x_test.abs()

#
# ~~~ Define models
torch.manual_seed(2025)
gpr = GPR(
        out_features = n_features,
        bws = ( torch.randn(n_features)**2 * torch.cdist(x_train,x_train).median().item() ).numpy().tolist(),
        scales = (torch.randn(n_features)**2).numpy().tolist(),
        etas = torch.linspace( 0.0001, 0.01, n_features ).numpy().tolist(),
        means = lambda x: torch.zeros( x.shape[0], n_features ).to( device=x.device, dtype=x.dtype ),
    )
_,_ = gpr.prior_mu_and_Sigma( x_train )
gpy = GPY(
        out_features = n_features,
        bws = copy(gpr.bws),
        scales = gpr.scales,
        etas = gpr.etas
    )

#
# ~~~ Run 'em
with support_for_progress_bars():
    for _ in trange(5):
        gpy.fit( x_train, y_train, verbose=False)
        mu_post_gpy, Sigma_post_gpy = gpy.post_mu_and_Sigma(x_test)
    for backend in (True,False):
        for _ in trange(5):
            gpr.fit( x_train, y_train, verbose=False, gpytorch=backend )
            mu_post, Sigma_post = gpr.post_mu_and_Sigma( x_test, gpytorch=backend )

print("")
print(f"Results are approximately the same, but not identical. Max absolute error {( mu_post - mu_post_gpy ).abs().max().item()}")
print("")

#
# ~~~ Check the results
i = 1
fig, axs = plt.subplots(1,2,figsize=(12,6))
for j,ax in enumerate(axs):
    mus   = mu_post if j==0 else mu_post_gpy
    ax.plot( x_test[:,i].cpu(), mus[:,i].cpu(), color="blue" )          # ~~~ plot the posterior mean
    ax.scatter( x_train[:,i].cpu(), y_train[:,i].cpu(), color="green" ) # ~~~ plot the training data
    ax.grid()
    ax.set_title("With GPyTorch" if torch.allclose(mus,mu_post_gpy) else "Without GPyTorch")

plt.tight_layout()
plt.show()
