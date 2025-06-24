import os

folder_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search"
)

#
# ~~~ Two different train/val splits of the same data
DATA = [
    "univar_missing_middle_normalized_12",
    "univar_missing_middle_normalized_12_cross_fold",
]

#
# ~~~ Primarily focus on pinning down good architectures
ARCHITECTURE = [
    #
    "univar_NN.univar_NN_30",  # ~~~ 1 hidden layer, 30 neurons
    "univar_NN.univar_NN_30_30",  # ~~~ 2 hidden layers, 30 neurons each
    "univar_NN.univar_NN_30_30_30",  # ~~~ 3 hidden layers, 30 neurons each
    "univar_NN.univar_NN_30_30_30_30",  # ~~~ 4 hidden layers, 30 neurons each
    #
    "univar_NN.univar_NN_100",  # ~~~ 1 hidden layer,  100 neurons
    "univar_NN.univar_NN_100_100",  # ~~~ 2 hidden layers, 100 neurons each
    "univar_NN.univar_NN_100_100_100",  # ~~~ 3 hidden layers, 100 neurons each
    "univar_NN.univar_NN_100_100_100_100",  # ~~~ 4 hidden layers, 100 neurons each
    #
    "univar_NN.univar_NN_250",  # ~~~ 1 hidden layer, 250 neurons
    "univar_NN.univar_NN_250_250",  # ~~~ 2 hidden layers, 250 neurons each
    "univar_NN.univar_NN_250_250_250",  # ~~~ 3 hidden layers, 250 neurons each
    "univar_NN.univar_NN_250_250_250_250",  # ~~~ 4 hidden layers, 250 neurons each
    #
    "univar_NN.univar_NN_500",  # ~~~ 1 hidden layer, 500 neurons
    "univar_NN.univar_NN_500_500",  # ~~~ 2 hidden layers, 500 neurons each
    "univar_NN.univar_NN_500_500_500",  # ~~~ 3 hidden layers, 500 neurons each
    "univar_NN.univar_NN_500_500_500_500",  # ~~~ 4 hidden layers, 500 neurons each
    #
]

#
# ~~~ Treat lr as a nuisance parameter, training it only barely enough to have confidence in which architectures are best
LR = [0.005, 0.001, 0.0005, 0.0001, 0.00001]

#
# ~~~ How many different neural networks to train for each architecture
N_TRIALS_PER_CONFIG = 5
