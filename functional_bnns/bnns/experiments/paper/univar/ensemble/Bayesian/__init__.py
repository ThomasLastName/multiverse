

import os
folder_name = os.path.join( os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search" )

#
# ~~~ Likelihood
LIKELIHOOD_STD = [ 0.1, 0.01, 0.001, 0.0001 ]

#
# ~~~ Training
WEIGHTING = [ "STANDARD", "NAIVE" ]
