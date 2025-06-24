import torch
from bnns.SSGE import SpectralSteinEstimator
import numpy as np
from quality_of_life.my_visualization_utils import *

torch.manual_seed(1234)
M = 3000  # ~~~ will be implicity rounded *down* the the nearest square number: int(sqrt(M))**2
eta = 0.0095
make_gif = False
D = 10
n_test = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

test_points = torch.randn(n_test, D, device=device)
samples = torch.randn(M, D, device=device)
xm = samples
x = test_points
self = SpectralSteinEstimator(eta=eta, samples=xm)


_xm = torch.cat((x, xm), dim=-2)
sigma = self.heuristic_sigma(_xm, _xm)
M = torch.tensor(xm.size(-2), dtype=torch.float)
Kxx, dKxx_dx = self.grad_gram(xm, xm, sigma)
if self.eta is not None:
    Kxx += self.eta * torch.eye(xm.size(-2), device=xm.device)


U, s, V = torch.linalg.svd(Kxx)
