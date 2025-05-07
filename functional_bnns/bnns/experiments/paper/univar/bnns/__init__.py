
import os
folder_name = os.path.join( os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search" )



### ~~~
## ~~~ Define the possible values of hyper-parameters not relating to the prior distribution
### ~~~

N_HYPERPAR_SAMPLES = 2

#
# ~~~ Architecture
ARCHITECTURE = [    # ~~~ == the list `BEST_4_ARCHITECTURES` defined in univar/dropout/process_results.py
        "univar_NN.univar_NN_30_30",
        "univar_NN.univar_NN_100_100",
        "univar_NN.univar_NN_250_250",
        "univar_NN.univar_NN_500_500_500_500"
    ]

#
# ~~~ Likelihood
LIKELIHOOD_STD = [ 0.1, 0.05, 0.01, 0.005, 0.001 ]

#
# ~~~ Choice of variational family
VARIATIONAL_FAMILY = [ "Normal", "Uniform" ]

#
# ~~~ Triaining
LR = [ 0.001, 0.0005, 0.0001, 0.00005, 0.00001 ]
FUNCTIONAL = [ True, False ]
MEASUREMENT_SET_SAMPLER = [ "random_points_only", "data_only", "current_batch_and_random_data" ]
N_MEAS    = [ 50,  200 ]
PRIOR_J   = [ 100, 500 ]
POST_J    = [ 25,  50  ]
PRIOR_ETA = [ 0.001, 0.1 ]
POST_ETA  = [ 0.001, 0.1 ]
PRIOR_M   = [ 500, 1500 ]
POST_M    = [ 50,   200  ]

