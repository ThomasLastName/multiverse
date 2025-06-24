import torch

torch.manual_seed(1234)
from bnns.utils.handling import convert_Tensors_to_Dataset

#
# ~~~ Data settings
n_train = 50
n_test = 1500
in_dim = 2
out_dim = 2
n_val = 30
noise = 0.2  # ~~~ pollute y_train wth Gaussian noise of variance noise**2
f = lambda x: torch.column_stack(
    [torch.cos(4 * torch.pi * x[:, 0]), torch.sin(2 * torch.pi * x[:, 1])]
)

#
# ~~~ Synthetic (noisy) training data
x_train = (
    2 * torch.rand(size=(n_train, in_dim)) ** 2 - 1
)  # ~~~ uniformly random points in [-1,1]
x_train = x_train.sign() * x_train.abs() ** (1 / 6)  # ~~~ push it away from zero
y_train = f(x_train) + noise * torch.randn(size=(n_train, out_dim))


#
# ~~~ Helper function
def make_uniform_data(n):
    x_test = torch.linspace(-1.5, 1.5, int((n) ** (1 / in_dim)))
    x_test = torch.cartesian_prod(x_test, x_test)
    x_test = torch.row_stack([x_test, 1.5 * torch.rand(size=(n - len(x_test), in_dim))])
    y_test = f(x_test)
    return x_test, y_test


#
# ~~~ Synthetic (noise-less) test data
x_test, y_test = make_uniform_data(n_test)

#
# ~~~ Synthetic (noisy) validation data
x_val, y_val = make_uniform_data(n_val)
y_val += noise * torch.randn(size=(n_val, out_dim))

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train, y_train)
D_test = convert_Tensors_to_Dataset(x_test, y_test)
D_val = convert_Tensors_to_Dataset(x_val, y_val)
