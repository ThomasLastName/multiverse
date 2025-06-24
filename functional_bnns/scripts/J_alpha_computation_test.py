import torch
from torch.func import jacrev, functional_call
from bnns.models.bivar_NN import NN
from bnns.utils import manual_Jacobian


batch_size = 15
number_of_input_features = NN[0].in_features
number_of_output_features = NN[-1].out_features  # ~~~ for slosh nets, 49719

#
# ~~~ Compute the Jacobian using only jacrev
torch.manual_seed(2024)
X = torch.rand(batch_size, number_of_input_features)
J_dict = jacrev(functional_call, argnums=1)(NN, dict(NN.named_parameters()), (X,))
J = torch.column_stack(
    [
        tens.reshape(number_of_output_features * batch_size, -1)
        for tens in J_dict.values()
    ]
)  # ~~~ shape will be ( out_featuers*batch_size, p ) where p is the total number of parameters in the whole model

#
# ~~~ Compute a sub-matrix of the same Jacobian using manual_Jacobian
V = X
for j in range(len(NN) - 1):
    V = NN[j](V)

final_J_manual = manual_Jacobian(V, number_of_output_features, bias=True)

#
# ~~~ Verify that they match
diff = final_J_manual - J[:, -(200 + number_of_output_features) :]
assert diff.abs().mean() == 0.0
