import os

folder_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search"
)


### ~~~
## ~~~ Define the possible values of hyper-parameters not relating to the prior distribution
### ~~~

N_HYPERPAR_SAMPLES = 2
SEED = [2021, 2022, 2023, 2024, 2025, 2026, 2027]  # ~~~ years of my PhD program ;)

#
# ~~~ Train/val split
DATA = [
    "univar_missing_middle_normalized_12",
    "univar_missing_middle_normalized_12_cross_fold",
]

#
# ~~~ Architecture
ARCHITECTURE = [  # ~~~ == the list `BEST_4_ARCHITECTURES` defined in univar/dropout/process_results.py
    "univar_NN.univar_NN_30_30",
    "univar_NN.univar_NN_100_100",
    "univar_NN.univar_NN_250_250",
    "univar_NN.univar_NN_500_500_500_500",
]

#
# ~~~ Likelihood
LIKELIHOOD_STD = [0.1, 0.01, 0.001, 0.0001]

#
# ~~~ Choice of variational family
VARIATIONAL_FAMILY = ["Normal", "Uniform", "Laplace"]

#
# ~~~ Triaining
LR = [0.001, 0.0005, 0.0001, 0.00001]
FUNCTIONAL = [True, False]
WEIGHTING = ["STANDARD", "NAIVE"]
MEASUREMENT_SET_SAMPLER = [
    "random_points_only",
    "data_only",
    "current_batch_and_random_data",
]
N_MEAS = [50, 200]
PRIOR_J = [100, 500]
POST_J = [25, 50]
PRIOR_ETA = [0.001, 0.1]
POST_ETA = [0.001, 0.1]
PRIOR_M = [500, 1500]
POST_M = [50, 200]
