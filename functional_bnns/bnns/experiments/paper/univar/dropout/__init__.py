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
# ~~~ Primarily focus on narrowing down good architectures

ARCHITECTURE = [
    "univar_NN.univar_NN_30_30_30_30",
    "univar_NN.univar_NN_30_30",
    "univar_NN.univar_NN_100_100",
    "univar_NN.univar_NN_100",
    "univar_NN.univar_NN_250_250",
    "univar_NN.univar_NN_250",
    "univar_NN.univar_NN_500_500_500_500",
    "univar_NN.univar_NN_500",
]

#
# ~~~ Treat lr as a nuisance parameter, training it only barely enough to have confidence in which architectures are best
LR = [0.001, 0.0001, 0.00001]

#
# ~~~ Treat dropout as a nuisance parameter, training it only barely enough to have confidence in which architectures are best
DROPOUT = [0.15, 0.3, 0.45, 0.6]
