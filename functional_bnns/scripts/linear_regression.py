import torch
import numpy as np
from matplotlib import pyplot as plt
from bnns.utils import lm, cor
from quality_of_life.my_plt_utils import abline

N = 300
x = torch.rand(N)
y = x**2 + 0.05 * torch.randn_like(x)

slope, intercept = lm(y, x)
slope_np, intercept_np = np.polyfit(x.numpy(), y.numpy(), 1)
assert np.isclose(slope_np, slope.item())
assert np.isclose(intercept_np, intercept.item())

r = cor(y, x)
r_torch = torch.corrcoef(torch.stack([x, y]))[0, 1]
assert torch.isclose(r_torch, r)


plt.scatter(x, y)
abline(slope, intercept)
plt.grid()
plt.tight_layout()
plt.show()
