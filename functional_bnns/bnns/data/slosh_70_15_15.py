
import os
import torch
import pyreadr                  # ~~~ from https://stackoverflow.com/a/61699417
import numpy as np
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset
from bnns.utils import process_grid_of_unit_cube
from bnns import __path__

#
# ~~~ Set path to the .rda file
PATH = os.path.join( __path__[0], "data", "slosh_dat_nj.rda" )

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
# ~~~ Package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)

#
# ~~~ First feaure: sea level rise
lower_sea_level_rise = -20
upper_sea_level_rise = 350

#
# ~~~ Second feaure: heading upon landfall
lower_landfall_heading = 204.0349
upper_landfall_heading = 384.02444

#
# ~~~ Third feaure: velocity upon landfall
lower_landfall_v = 0
upper_landfall_v = 40

#
# ~~~ Fourth feaure: minimum air pressure upon landfall
lower_min_landfall_p = 930
upper_min_landfall_p = 980

#
# ~~~ Fifth feaure: latitude upon landfall
lower_landfall_lat = 38.32527
upper_landfall_lat = 39.26811

#
# ~~~ Finally, generate a relatively "fine" grid of the input domain (in 5D, no reasonably sized grid is really fine)
# grid = torch.quasirandom.SobolEngine(dimension=5).draw(25000) # ~~~ a non-random, "space filling grid"
grid_of_unit_cube = torch.rand(25000,5)  # ~~~ a uniform random grid
bounds = torch.tensor([
        [ lower_sea_level_rise, upper_sea_level_rise ],
        [ lower_landfall_heading, upper_landfall_heading ],
        [ lower_landfall_v, upper_landfall_v ],
        [ lower_min_landfall_p, upper_min_landfall_p ],
        [ lower_landfall_lat, upper_landfall_lat ]
    ])
extrapolary_grid, interpolary_grid = process_grid_of_unit_cube( grid_of_unit_cube, bounds )

interpolary_grid = torch.cat([ x_train + 0.1*torch.randn_like(x_train).sign()*torch.rand_like(x_train) for _ in range(5) ])
dists = torch.cdist(interpolary_grid,x_train).min(dim=1).values
