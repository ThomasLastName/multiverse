
import torch
from torch import nn
from matplotlib import pyplot as plt
import seaborn as sns
import bnns
from bnns.utils import log_gaussian_pdf

torch.manual_seed(2025)
mixture_prior = bnns.MixtureWeightPrior2015BNN( nn.Linear(1,1,bias=False) )
samples = mixture_prior.prior_forward( torch.tensor([[1.]]), n=5000 ).detach().flatten()
sns.kdeplot(samples)
x = torch.linspace(-3,3,540)
plt.plot( x, mixture_prior.pi*log_gaussian_pdf( x, torch.zeros_like(x), mixture_prior.sigma1, multivar=False ).exp() )
plt.plot( x, (1-mixture_prior.pi)*log_gaussian_pdf( x, torch.zeros_like(x), mixture_prior.sigma2, multivar=False ).exp() )
plt.show()
