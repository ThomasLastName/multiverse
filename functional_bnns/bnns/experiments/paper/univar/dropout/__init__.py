
import os
folder_name = os.path.join( os.path.dirname(os.path.abspath(__file__)), "hyperparameter_search" )

#
# ~~~ Two different train/val splits of the same data
DATA = [
        "univar_missing_middle_normalized_12",
        "univar_missing_middle_normalized_12_cross_fold"
    ]

#
# ~~~ Primarily focus on narrowing down good architectures
ARCHITECTURE = [    # ~~~ == the list `BEST_12_ARCHITECTURES` defined in univar/det_nn/process_results.py
        #
        "univar_NN.univar_NN_30_30",              # ~~~ 2 hidden layers, 30 neurons each
        "univar_NN.univar_NN_30_30_30_30",          # ~~~ 4 hidden layers, 30 neurons each
        #
        "univar_NN.univar_NN_100",              # ~~~ 1 hidden layer,  100 neurons
        "univar_NN.univar_NN_100_100",            # ~~~ 2 hidden layers, 100 neurons each
        #
        "univar_NN.univar_NN_250",              # ~~~ 1 hidden layer, 250 neurons
        "univar_NN.univar_NN_250_250",            # ~~~ 2 hidden layers, 250 neurons each
        #
        "univar_NN.univar_NN_500",              # ~~~ 1 hidden layer, 500 neurons
        "univar_NN.univar_NN_500_500_500_500",      # ~~~ 4 hidden layers, 500 neurons each
        #
        "univar_NN.univar_NN_750",              # ~~~ 1 hidden layer, 750 neurons
        "univar_NN.univar_NN_750_750",            # ~~~ 2 hidden layers, 750 neurons each
        #
        "univar_NN.univar_NN_1000",             # ~~~ 1 hidden layer, 1000 neurons
        "univar_NN.univar_NN_1000_1000",          # ~~~ 2 hidden layers, 1000 neurons each
    ]

#
# ~~~ Treat lr as a nuisance parameter, training it only barely enough to have confidence in which architectures are best
LR = [ 0.001, 0.0005, 0.0001, 0.00001 ]

#
# ~~~ Treat dropout as a nuisance parameter, training it only barely enough to have confidence in which architectures are best
DROPOUT = [ 0.15, 0.3, 0.45, 0.6 ]
