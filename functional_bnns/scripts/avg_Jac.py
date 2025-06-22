
import torch
from tqdm import trange
from bnns.utils import support_for_progress_bars

M = 1000
d = 400
x = torch.randn(M,d)
sigma = 1.

K = torch.exp( -(torch.cdist(x,x)/sigma)**2/2 )

#
# ~~~ Original implementation (einsum is faster but has same memory footprint)
diff = (x.unsqueeze(-2) - x.unsqueeze(-3)) / (sigma**2) # [M x M x d]
K_Jacobians = K.unsqueeze(-1) * (-diff)
avg_Jac = K_Jacobians.mean(dim=0)

#
# ~~~ Compute the average using a loop, in order to never create any third order tensor (for memory's sake)
iterative_avg_Jac = torch.zeros(M, d)
with support_for_progress_bars():
    for i in trange(d):
        diff_i = (x[:, i].unsqueeze(-1) - x[:, i].unsqueeze(-2)) / sigma**2  # [M x M]
        K_Jacobian_i = K * (-diff_i)
        iterative_avg_Jac[:, i] = K_Jacobian_i.mean(dim=0)

assert torch.equal( avg_Jac, iterative_avg_Jac )
