
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from tqdm import trange

#
# ~~~ The guts of the model
from bnns.Stein_GD import SequentialSteinEnsemble as Ensemble

#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, set_Dataset_attributes, generate_json_filename

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_base_utils  import support_for_progress_bars
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list


### ~~~
## ~~~ Config/setup
### ~~~

STEIN = True
VMAP = False

SEED = 2024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from bnns.models.bivar_NN import NN
from bnns.data.bivar_trivial import D_train as data
X, y = data.X.to(DEVICE), data.y.to(DEVICE)
NN = NN.to(DEVICE)



### ~~~
## ~~~ Do a Stein neural network ensemble
### ~~~

for OLD_TRAINING in [True,False]:
    #
    # ~~~ Set the seed
    _ = torch.manual_seed(SEED)
    #
    # ~~~ Instantiate an ensemble (the same both times)
    ensemble = Ensemble(
            architecture = nonredundant_copy_of_module_list(NN),    # ~~~ copy for reproducibility
            n_copies = 100,
            Optimizer = lambda params: torch.optim.Adam( params, lr=0.001 ),
            conditional_std = torch.tensor(0.19),
            device = DEVICE
        )
    #
    # ~~~ Test the use of vmap in the __call__ method
    assert ( ensemble(X,vectorized=True) - ensemble(X,vectorized=False) ).abs().max() < 1e-6
    if OLD_TRAINING: # ~~~ only run this test for one of the two training methods (it is the same in both cases)
        print("")
        print("    Testing the speed of the __call__ method with and without vmap.")
        print("")
        with support_for_progress_bars():
            for _ in trange( 100, desc="With vmap" ):
                _ = ensemble(X,vectorized=True)
            for _ in trange( 100, desc="Without vmap" ):
                _ = ensemble(X,vectorized=False)
    #
    # ~~~ Test that the vmap is using the upddated parameters (which vectorized=False certainly does)
    ensemble.train_step( X, y, stein=STEIN, easy_implementation=OLD_TRAINING, vectorized=VMAP )
    assert ( ensemble(X,vectorized=True) - ensemble(X,vectorized=False) ).abs().max() < 1e-6
    if OLD_TRAINING:
        result_of_old_training = ensemble(X)
        # grad_from_old_training = get_flat_grads(ensemble.models[0])
    else:
        result_of_new_training = ensemble(X)
        # grad_from_new_training = get_flat_grads(ensemble.models[0])

#
# ~~~ Test the same update was performed whether using the original or the new implementation
assert torch.allclose( result_of_new_training, result_of_old_training )
print("")
print("    Testing the speed of the train_step method with and without einsum.")
print("")
with support_for_progress_bars():
    for _ in trange( 20, desc="With einsum" ):
        ensemble.train_step( X, y, stein=STEIN, easy_implementation=False )
    for _ in trange( 20, desc="Without einsum" ):
        ensemble.train_step( X, y, stein=STEIN, easy_implementation=True )