### ~~~
## ~~~ Import block
### ~~~

import torch
from tqdm import trange
from bnns.Ensemble import SequentialSteinEnsemble as Ensemble
from bnns.utils import support_for_progress_bars, nonredundant_copy_of_module_list


### ~~~
## ~~~ Config/setup
### ~~~

STEIN = True
N_COPIES = 100

SEED = 2024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from bnns.models.bivar_NN import NN
from bnns.data.bivar_trivial import D_train as data

X, y = data.X.to(DEVICE), data.y.to(DEVICE)
NN = NN.to(DEVICE)


### ~~~
## ~~~ Test all three forward passes to confirm they are equivalent, and compare their speeds
### ~~~

#
# ~~~ Set the seed
_ = torch.manual_seed(SEED)

#
# ~~~ Instantiate an ensemble (the same both times)
ensemble = Ensemble(
    architecture=nonredundant_copy_of_module_list(NN),  # ~~~ copy for reproducibility
    n_copies=N_COPIES,
    Optimizer=lambda params: torch.optim.Adam(params, lr=0.001),
    likelihood_std=torch.tensor(0.19),
    device=DEVICE,
)

with torch.no_grad():
    assert (ensemble(X, method="vmap") - ensemble(X, method="bmm")).abs().max() < 1e-6
    assert (ensemble(X, method="vmap") - ensemble(X, method="naive")).abs().max() < 1e-6
    print("")
    print("    Testing the speed of the forward pass.")
    print("")
    with support_for_progress_bars():
        for method in ["naive", "bmm", "vmap"]:
            for _ in trange(100, desc=f"method={method}"):
                _ = ensemble(X, method=method)


### ~~~
## ~~~ Test the naive training method against the current best
### ~~~

for OLD_TRAINING in [True, False]:
    #
    # ~~~ Set the seed
    _ = torch.manual_seed(SEED)
#     #
#     # ~~~ Instantiate an ensemble (the same both times)
#     ensemble = Ensemble(
#             architecture = nonredundant_copy_of_module_list(NN),    # ~~~ copy for reproducibility
#             n_copies = N_COPIES,
#             Optimizer = lambda params: torch.optim.Adam( params, lr=0.001 ),
#             likelihood_std = torch.tensor(0.19),
#             device = DEVICE
#         )
#     #
#     # ~~~ Test that the vmap is using the upddated parameters (which vectorized=False certainly does)
#     _ = ensemble.train_step( X, y, stein=STEIN, naive_implementation=OLD_TRAINING, vectorized_forward=(not OLD_TRAINING) )
#     if OLD_TRAINING:
#         result_of_old_training = ensemble(X)
#     else:
#         result_of_new_training = ensemble(X)

# #
# # ~~~ Test the same update was performed whether using the original or the new implementation
# assert ( result_of_new_training - result_of_old_training ).abs().mean() < 1e-6
# print("")
# print("    Testing the speed of the train_step method with and without einsum.")
# print("")
# with support_for_progress_bars():
#     for _ in trange( 10, desc="Original training implementation" ):
#         _ = ensemble.train_step( X, y, stein=STEIN, naive_implementation=True, vectorized_forward=False )
#     for _ in trange( 10, desc="Optimized training implementation" ):
#         _ = ensemble.train_step( X, y, stein=STEIN, naive_implementation=False, vectorized_forward=True )
