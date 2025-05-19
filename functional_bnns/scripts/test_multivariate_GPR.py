
import torch
from bnns.GPR import GPY as GPR
# from bnns.GPR import simple_mean_zero_RPF_kernel_GP as GPR
from bnns.data.bivar_trivial import x_train, y_train, x_test, y_test
from matplotlib import pyplot as plt

#
# ~~~ Do GPR
torch.set_default_dtype(torch.float64)
# gpr = GPR( out_features=2, bandwidths=[0.2,0.5], scales=[1,1.1], etas=[0.0001,0.1] )
gpr = GPR( out_features=2, etas=[0.0001,0.0001] )
gpr.fit( x_train, y_train )
mu_post, sigma_post = gpr.post_mu_and_Sigma(x_test)
predictions = gpr( x_test, n=100 )

#
# ~~~ Check the results
fig, axs = plt.subplots(1,2,figsize=(12,6))
for j,ax in enumerate(axs):
    ax.plot( x_test[:,j], mu_post[:,j], color="blue" )
    for p in predictions: _ = ax.plot( x_test[:,j], p[:,j], color="blue", alpha=0.1 )
    ax.plot( x_test[:,j], y_test[:,j], linestyle="--", color="green" )
    ax.scatter( x_train[:,j], y_train[:,j], color="green" )
    ax.set_ylim([-6,6])
    ax.grid()

plt.tight_layout()
plt.show()
