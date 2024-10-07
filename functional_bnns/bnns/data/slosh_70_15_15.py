
import os
import torch
import pyreadr                  # ~~~ from https://stackoverflow.com/a/61699417
import numpy as np
from quality_of_life.my_base_utils import find_root_dir_of_repo
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset

#
# ~~~ Set path to the .rda file
root = find_root_dir_of_repo()
PATH = os.path.join( root, "functional_bnns", "bnns", "data", "slosh_dat_nj.rda" )
if __name__ == "__main__":
    ans = input(f"    Is the path {PATH} correct?\n    Enter 'y' for yes, any other key for no.\n")
    if not ans.lower()=="y":
        PATH = input("    Please type the path without quotes and press enter:\n") # ~~~ e.g., /Users/winckelman/Downloads/slosh_dat_nj.rda

#
# ~~~ Extract the data as numpy arrays
DATA = pyreadr.read_r(PATH)     # ~~~ from https://stackoverflow.com/a/61699417
coords_np  =  DATA["coords"].to_numpy()
inputs_np  =  DATA["inputs"].to_numpy()
out_np     =     DATA["out"].to_numpy()

#
# ~~~ Compute indices for a train/val/test split (same code as in slosh_70_15_15_centered_pca.py and slosh_70_15_15_standardized_pca.py)
np.random.seed(2024)
n_train = 2600
n_test = 700
n_val = 700
n = len(inputs_np)
assert len(inputs_np) == 4000 == len(out_np)
assert n_train + n_test + n_val == n
idx = np.random.permutation(n)
idx_train, idx_test, idx_val = np.split( idx, [n_train,n_train+n_test] )

#
# ~~~ Use indices for a train/val/test split
x_train = torch.from_numpy(inputs_np[idx_train])
x_test = torch.from_numpy(inputs_np[idx_test])
x_val = torch.from_numpy(inputs_np[idx_val])
y_train = torch.from_numpy(out_np[idx_train])
y_test = torch.from_numpy(out_np[idx_test])
y_val = torch.from_numpy(out_np[idx_val])

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)

