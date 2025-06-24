import torch
from torch.func import jacrev, functional_call
from bnns.models.bivar_NN import NN

#
# ~~~ Make up some data
torch.manual_seed(2024)
x = torch.randn(50, 2)
batch_size = x.shape[0]

#
# ~~~ Compute the full Jacobian using torch.func (second example at https://pytorch.org/docs/stable/func.migrating.html#functorch-make-functional)
Jacobians = jacrev(functional_call, argnums=1)(
    NN, dict(NN.named_parameters()), (x,)
)  # ~~~ a dictionary with the same keys as NN.named_parameters()
auto_J = Jacobians["4.weight"]
assert auto_J.shape == (50, 2, 2, 100)
auto_J = auto_J.reshape(100, 200)  # 2*100 parameters -> 2*50 predictions

#
# ~~~ Construct the Jacobian using the values from immediately before the final linear transformation
v = x
for j in range(len(NN) - 1):
    v = NN[j](v)

pencil_J = torch.row_stack(
    [
        torch.row_stack(
            [
                torch.concat([v[j], torch.zeros_like(v[j])]),
                torch.concat([torch.zeros_like(v[j]), v[j]]),
            ]
        )
        for j in range(batch_size)
    ]
)

#
# ~~~ Observe that the two are the same
assert auto_J.shape == pencil_J.shape
assert torch.allclose(auto_J, pencil_J)

#
# ~~~ Save memory by computiung the Jacobian only with respect to the very final layer's parameters
final_J = jacrev(functional_call, argnums=1)(
    NN[-1], dict(NN[-1].named_parameters()), (v,)
)
final_J = final_J["weight"].reshape(100, 200)
assert torch.allclose(auto_J, final_J)
