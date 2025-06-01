
import os
import torch
folder_name = os.path.join( os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search" )

#
# ~~~ Training
from bnns.data.univar_missing_middle import x_train
X = x_train.reshape(-1,1)
median_scale = torch.cdist(X,X).median().item()
BW = [ median_scale/4, median_scale/2, None, 2*median_scale, 4*median_scale ]
