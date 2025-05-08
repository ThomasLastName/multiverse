
import os
from math import exp
folder_name = os.path.join( os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search" )

#
# ~~~ Hyper-parameters of the prior distribution
PI     = [   1/3,     2/3   ]   # ~~~ hyper-parameter of the mixture prior
SIGMA1 = [ exp(-0), exp(-2) ]   # ~~~ hyper-parameter of the mixture prior
SIGMA2 = [ exp(-6), exp(-8) ]   # ~~~ hyper-parameter of the mixture prior
