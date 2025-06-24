from torch.func import jacrev, functional_call
from bnns.models.tiny_slosh_NN import NN
from bnns.data.slosh_70_15_15 import D_train

x_train = D_train.X.float()
x = x_train[:100]

#
# ~~~ On my LANL mac, this results in `zsh: killed` however, on my PC (using x=torch.randn(100,5)), I get the more informative error `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3725.29 GiB`
jacobians = jacrev(functional_call, argnums=1)(
    NN, dict(NN.named_parameters()), (x,)
)  # ~~~ a dictionary with the same keys as NN.named_parameters()
