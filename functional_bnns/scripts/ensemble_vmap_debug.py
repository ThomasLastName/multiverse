import torch
from torch import nn, func, vmap
import copy
from bnns.utils import nonredundant_copy_of_module_list


### ~~~
## ~~~ Define the ensemble class
### ~~~

loss_fn = nn.MSELoss()


class SteinEnsemble:
    #
    # ~~~
    def __init__(self, list_of_NNs, Optimizer):
        #
        # ~~~ Establish the basic attributes
        self.models = (
            list_of_NNs  # ~~~ each "particle" is (the parameters of) a neural network
        )
        self.n_models = len(self.models)
        self.optimizers = [Optimizer(model.parameters()) for model in self.models]
        #
        # ~~~ Weird stuff for vmap, which I mindlessly copied from https://pytorch.org/tutorials/intermediate/ensembling.html
        base_model = copy.deepcopy(self.models[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return func.functional_call(base_model, (params, buffers), (x,))

        self.fmodel = fmodel  # ~~~ vmap requires this
        self.params, self.buffers = func.stack_module_state(self.models)

    #
    def train_step(self, X, y, vectorized=True, zero_out_grads=True):
        #
        # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
        if vectorized:
            losses = torch.stack([loss_fn(model(X), y) for model in self.models])
            losses.sum().backward()
        else:
            for model in self.models:
                loss = loss_fn(model(X), y)
                loss.backward()
        #
        # ~~~ Do the update
        for optimizer in self.optimizers:
            optimizer.step()
            if zero_out_grads:
                optimizer.zero_grad()
        #
        # ~~~ As far as I can tell, the params used by vmap need to be updated manually like this
        with torch.no_grad():
            self.params, self.buffers = func.stack_module_state(self.models)

    #
    # ~~~ forward for the full ensemble
    def __call__(self, X, vectorized=True):
        if vectorized:
            return vmap(self.fmodel, in_dims=(0, 0, None))(self.params, self.buffers, X)
        else:
            return torch.stack([model(X) for model in self.models])


class Ensemble(SteinEnsemble):
    def __init__(self, architecture, n_copies, device="cpu", *args, **kwargs):
        self.device = device
        super().__init__(
            list_of_NNs=[
                nonredundant_copy_of_module_list(architecture, sequential=True).to(
                    device
                )
                for _ in range(n_copies)
            ],
            *args,
            **kwargs
        )


### ~~~
## ~~~ Config/setup
### ~~~

seed = 2024
device = "cuda" if torch.cuda.is_available() else "cpu"
n_Stein_particles = 100
lr = 0.001

Optimizer = torch.optim.Adam
from bnns.models.bivar_NN import NN
from bnns.data.bivar_trivial import D_train as data

X, y = data.X.to(device), data.y.to(device)
NN = NN.to(device)


def make_ensemble():
    torch.manual_seed(seed)
    return Ensemble(
        architecture=nonredundant_copy_of_module_list(
            NN
        ),  # ~~~ copy for reproducibility
        n_copies=n_Stein_particles,
        Optimizer=lambda params: Optimizer(params, lr=lr),
        device=device,
    )


### ~~~
## ~~~ Run tests
### ~~~

#
# ~~~ Test reproducibility
for bool in [True, False]:
    ensemble = make_ensemble()
    ensemble.train_step(X, y, vectorized=bool)
    first = ensemble(X)
    ensemble = make_ensemble()
    ensemble.train_step(X, y, vectorized=bool)
    second = ensemble(X)
    assert torch.allclose(first, second)

#
# ~~~ Test compatibility
ensemble = make_ensemble()
ensemble.train_step(X, y, vectorized=True)
first = ensemble(X)
ensemble = make_ensemble()
ensemble.train_step(X, y, vectorized=False)
second = ensemble(X)
assert torch.allclose(first, second)

#
# ~~~ Test non-triviality
ensemble = make_ensemble()
first = ensemble(X)
ensemble.train_step(X, y, vectorized=False)
second = ensemble(X)
assert not torch.allclose(first, second)
