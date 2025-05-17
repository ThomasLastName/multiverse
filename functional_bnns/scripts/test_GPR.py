
import torch
from bnns.GPR import simple_mean_zero_RPF_kernel_GP as GPR
from matplotlib import pyplot as plt

#
# ~~~ Make up data
torch.manual_seed(2025)
x_train = torch.linspace(-1,1,5).reshape(-1,1)
y_train = x_train.abs() + 0.05*torch.randn_like(x_train)
x_test = torch.linspace(-1,1,301).reshape(-1,1)
y_test = x_test.abs()

#
# ~~~ Do GPR
gpr = GPR( out_features=1 )
gpr.fit( x_train, y_train )
mu_post, sigma_post = gpr.post_mu_and_Sigma(x_test)
predictions = gpr( x_test, n=100 )

#
# ~~~ Check the results
plt.plot( x_test, mu_post, color="blue" )
for p in predictions: plt.plot( x_test, p, color="blue", alpha=0.1 )

plt.plot( x_test, y_test, linestyle="--", color="green" )
plt.scatter( x_train, y_train, color="green" )
plt.grid()
plt.tight_layout()
plt.show()
