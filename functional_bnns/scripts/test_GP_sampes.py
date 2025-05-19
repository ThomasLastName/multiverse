
import torch
from matplotlib import pyplot as plt
from bnns.utils import randmvns

#
# ~~~ Data
x = torch.linspace(-1,1,101).reshape(-1,1)

#
# ~~~ Kernel hyper-parameters
a = 1.
bw = 1
eta = 1e-4

#
# ~~~ Kernel matrix
K = a*torch.exp(-torch.cdist(x,x)**2/2/bw) + eta*torch.diag(torch.ones_like(x).flatten())

#
# ~~~ Sample from the GP
n_samples = 5
GP = randmvns( torch.zeros_like(x), torch.linalg.cholesky(K).unsqueeze(0), n_samples )
# GP = (torch.linalg.cholesky(K)@torch.randn( K.shape[0], n_samples )).T
for p in GP: _=plt.plot( x, p, color="blue", alpha=0.3 )

plt.grid()
plt.tight_layout()
plt.show()




