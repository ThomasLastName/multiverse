
import numpy as np
import torch
import os
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset
from bnns.data.slosh_70_15_15 import coords_np, inputs_np, out_np, idx_train, idx_test, idx_val, extrapolary_grid, interpolary_grid, data_folder

#
# ~~~ Generate U, s, and V
avg_out = np.mean(out_np,axis=0)
try:
    #
    # ~~~ Load the processed data
    U = torch.load(os.path.join( data_folder, "slosh_centered_U.pt"))
    s = torch.load(os.path.join( data_folder, "slosh_centered_s.pt"))
    V = torch.load(os.path.join( data_folder, "slosh_centered_V.pt"))
except:
    #
    # ~~~ Load the unprocessed data
    data_matrix = torch.from_numpy( out_np - avg_out )
    #
    # ~~~ Process the data (do SVD)
    print("Computing principal components...")
    torch.manual_seed(2024)     # ~~~ torch.svd_lowrank is stochastic
    U, s, Vt = torch.linalg.svd( data_matrix, full_matrices=False )
    V = Vt.T
    #
    # ~~~ Save the processed data
    torch.save( U, os.path.join( data_folder, "slosh_centered_U.pt" ))
    torch.save( s, os.path.join( data_folder, "slosh_centered_s.pt" ))
    torch.save( V, os.path.join( data_folder, "slosh_centered_V.pt" ))

#
# ~~~ Determine how many principal components `r` are needed to explain 99% of the variance
percentage_of_variance_explained = (s**2).cumsum(dim=0) / (s**2).sum()
r = (percentage_of_variance_explained<.99).int().argmin().item()    # ~~~ the first index at which percentage_of_variance_explained>=.99
U_truncated = U[:,:r]
s_truncated = s[:r]
V_truncated = V[:,:r]

#
# ~~~ Use indices for a train/val/test split
x_train = torch.from_numpy(inputs_np[idx_train])
x_test = torch.from_numpy(inputs_np[idx_test])
x_val = torch.from_numpy(inputs_np[idx_val])
y_train = U_truncated[idx_train]
y_test = U_truncated[idx_test]
y_val = U_truncated[idx_val]
unprocessed_y_train = torch.from_numpy(out_np[idx_train])
unprocessed_y_test = torch.from_numpy(out_np[idx_test])
unprocessed_y_val = torch.from_numpy(out_np[idx_val])
avg_out = torch.from_numpy(avg_out)

#
# ~~~ Sanity check
reconstructed_y_train = y_train @ s_truncated.diag() @ V_truncated.T + avg_out
assert (( reconstructed_y_train - unprocessed_y_train)**2).mean() < 0.02

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
