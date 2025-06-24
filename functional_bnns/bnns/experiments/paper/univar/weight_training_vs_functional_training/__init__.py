import os
from math import exp

folder_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search"
)

#
# ~~~ Architecture
ARCHITECTURE = [  # ~~~ == the list `BEST_4_ARCHITECTURES` defined in univar/dropout/process_results.py
    "univar_NN.univar_NN_30_30",
    "univar_NN.univar_NN_100_100",
    "univar_NN.univar_NN_250_250",
    "univar_NN.univar_NN_500_500_500_500",
]
VARIATIONAL_FAMILY = ["Normal", "Uniform"]


#
# ~~~ Likelihood
LIKELIHOOD_STD = [0.1, 0.01, 0.001]

#
# ~~~ Prior
MODEL = [
    "MixtureWeightPrior2015BNN",  # ~~~ mixture prior proposed in Blundell et al. 2015 (https://arxiv.org/abs/1505.05424)
    "FullSupportLocScaleBNN",  # ~~~ fully factorized Gaussian prior which as in torchbnn
]
PI = [1 / 4, 2 / 4, 3 / 4]  # ~~~ hyper-parameter of the mixture prior
SIGMA1 = [exp(-0), exp(-1), exp(-2)]  # ~~~ hyper-parameter of the mixture prior
SIGMA2 = [exp(-6), exp(-7), exp(-8)]  # ~~~ hyper-parameter of the mixture prior
PRIOR_TYPE = [
    "torch.nn.init",
    "Tom",
    "IID",
]  # ~~~ hyper-parameter of the Gaussian prior
SCALE = [0.25, 1, 4]  # ~~~ hyper-parameter of the Gaussian prior

#
# ~~~ Triaining
LR = [0.001, 0.0001]
FUNCTIONAL = [True, False]
PROJECTION_METHOD = ["HARD", "Blundell", "torchbnn"]
DEFAULT_INITIALIZATION = [None, "new", "old"]
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
