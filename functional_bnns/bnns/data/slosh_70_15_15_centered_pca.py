
import numpy as np
import torch
import os
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset
from quality_of_life.my_base_utils import find_root_dir_of_repo
from bnns.data.slosh_70_15_15 import coords_np, inputs_np, out_np, idx_train, idx_test, idx_val

#
# ~~~ Establish the path to the folder `bnns/data`
root = find_root_dir_of_repo()
PATH = os.path.join( root, "functional_bnns", "bnns", "data" )

#
# ~~~ Generate U, s, and V
try:
    #
    # ~~~ Load the processed data
    U = torch.load(os.path.join( PATH, "slosh_centered_U.pt"))
    s = torch.load(os.path.join( PATH, "slosh_centered_s.pt"))
    V = torch.load(os.path.join( PATH, "slosh_centered_V.pt"))
except:
    #
    # ~~~ Load the unprocessed data
    data_matrix = torch.from_numpy( out_np - np.mean(out_np,axis=0) )
    #
    # ~~~ Process the data (do SVD)
    torch.manual_seed(2024)     # ~~~ torch.svd_lowrank is stochastic
    U, s, Vt = torch.linalg.svd( data_matrix, full_matrices=False )
    V = Vt.T
    #
    # ~~~ Save the processed data
    torch.save( U, os.path.join( PATH, "slosh_centered_U.pt" ))
    torch.save( s, os.path.join( PATH, "slosh_centered_s.pt" ))
    torch.save( V, os.path.join( PATH, "slosh_centered_V.pt" ))

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

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
