
import torch
from bnns.data.univar_missing_middle import x_train, y_train, x_val, y_val, x_test, y_test, n_train, n_val
from bnns.data.univar_missing_middle import f, grid, interpolary_grid, extrapolary_grid, random_points_only, data_only, current_batch_and_random_data
from bnns.utils.handling import convert_Tensors_to_Dataset

#
# ~~~ Consider the training and validation sets together
x_non_test = torch.concatenate([ x_train, x_val ])
y_non_test = torch.concatenate([ y_train, y_val ])

#
# ~~~ We could recover the original train/val split by defining x_train=x_non_test[:n_train] and x_val=x_non_test[n_train:]
assert torch.equal( x_train, x_non_test[:n_train] )
assert torch.equal( x_val,   x_non_test[n_train:] )

#
# ~~~ We will instead create a new train/val split of the same data by defining x_train=x_non_test[:n_val] and x_val=x_non_test[n_val:]
x_val   = x_non_test[:n_val]    # ~~~ the first `n_val` entries of x_non_test
y_val   = y_non_test[:n_val]
x_train = x_non_test[n_val:]    # ~~~ the remaining `n_train` entries of x_non_test
y_train = y_non_test[n_val:]
assert  len(x_val)  ==  n_val  ==  len(y_val)
assert len(x_train) == n_train == len(y_train)

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)