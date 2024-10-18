
import torch

from bnns.models.bivar_NN import NN
from bnns.data.bivar_trivial import x_train
from bnns.Ensemble import SequentialSteinEnsemble as Ensemble

ensemble = Ensemble(
    architecture = NN,
    n_copies = 1000,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    Optimizer = torch.optim.Adam,
    conditional_std = 0.1,
    bw = 0.1
)

x_train = x_train.to(ensemble.devive)
ensemble(x_train)
