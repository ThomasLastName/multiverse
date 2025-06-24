import copy
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from torch import vmap


#
# ~~~ Setup
VMAP = False
torch.manual_seed(0)
device = "cuda"
num_models = 10
X = torch.randn(64, 1, 28, 28, device=device)
y = torch.randn(10, 64, 10, device=device)
models = [
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    for _ in range(num_models)
]


#
# ~~~ Setup for vmap (!!I don't understand this code, I just copied it from https://pytorch.org/tutorials/intermediate/ensembling.html)
params, buffers = stack_module_state(models)
base_model = copy.deepcopy(models[0])
base_model = base_model.to("meta")


def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))


#
# ~~~ Compute losses
if VMAP:
    #
    # ~~~ Do the forward pass with vmap
    predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, X)
    loss_vmap = ((predictions_vmap - y) ** 2).mean()
    loss_vmap.backward()
    predictions = torch.stack([model(X) for model in models])  # ~~~ for comparison
else:
    #
    # ~~~ Do the forward pass without vmap
    predictions = torch.stack([model(X) for model in models])
    loss = ((predictions - y) ** 2).mean()
    loss.backward()
    predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(
        params, buffers, X
    )  # ~~~ for comparison

assert torch.allclose(predictions_vmap, predictions, atol=1e-3, rtol=1e-5)
assert models[0][1].weight.grad is not None
