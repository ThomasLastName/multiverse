
import torch
from torch.func import jacrev, functional_call
from bnns.models.bivar_NN import NN     # ~~~ memory overflow if you swap `bivar_NN` for `tiny_slosh_NN`
from bnns.utils import manual_Jacobian
from tqdm import trange

#
# ~~~ Make up some data
batch_size = 15
number_of_input_features  =  NN[0].in_features
number_of_output_features = NN[-1].out_features # ~~~ for slosh nets, 49719

#
# ~~~ Test how fast manual computation is
torch.manual_seed(2024)
for _ in trange(100, desc="Using a handwritten formula" ):
    V = torch.randn( batch_size, number_of_input_features )
    final_J_manual = manual_Jacobian( NN[:-1](V), number_of_output_features )


#
# ~~~ Test how fast torch.func is
torch.manual_seed(2024)
for _ in trange(100, desc="Using torch.func.jacrev" ):
    V = torch.randn( batch_size, number_of_input_features )
    for j in range(len(NN)-1):
        V = NN[j](V)
    final_J = jacrev(functional_call, argnums=1)( NN[-1], dict(NN[-1].named_parameters()), (V,) )
    final_J = final_J["weight"].reshape(number_of_output_features*batch_size,-1)

assert torch.allclose( final_J, final_J_manual )
