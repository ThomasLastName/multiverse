import torch
from bnns.SSGE import SpectralSteinEstimator
from matplotlib import pyplot as plt


device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)
LB = -5
UB = 5
x = torch.linspace(LB, UB, 150, device=device).view(-1, 1)
M = 100
eta = 0.0095


ground_truth = lambda x: -x
score = ground_truth(x)
samples = torch.randn(M, 1, device=device)
try:
    score_estimator = SpectralSteinEstimator(samples=samples, eta=eta)
except RuntimeError:
    print("Using eig instead of eigh due to the pytorch source code bug.")
    score_estimator = SpectralSteinEstimator(samples=samples, eta=eta, h=False)

est_score = score_estimator(x)


plt.figure()
x = x.cpu()
plt.plot(x, score.cpu(), lw=2, label=r"$\nabla_x \log(x)$")
plt.plot(x, est_score.cpu(), lw=2, label=r"$\hat{\nabla}_x \log(x)$")
plt.plot(x, -(x**2) / 2, lw=2, label=r"$\log(x)$")
plt.title(
    f"Gaussian Distribution with {M} samples with " + r"$\eta$" + f" = {eta}",
    fontsize=15,
)
plt.legend(fontsize=15)
plt.show()
