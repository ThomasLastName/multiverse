

import os
import torch
import numpy as np
from quality_of_life.my_base_utils import dict_to_json
from bnns.experiments.paper.univar_det_nn import folder_name



### ~~~
## ~~~ Create a folder and populate it with a whole bunch of JSON files
### ~~~

#
# ~~~ Define all hyperparmeters *including* even the ones not to be tuned
EXPLORE_DURING_TUNING = "placeholder value"
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE" : "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE" : "float",
    "SEED" : 2024,
    #
    # ~~~ Which problem
    "DATA" : EXPLORE_DURING_TUNING,
    "MODEL" : EXPLORE_DURING_TUNING,
    #
    # ~~~ For training
    "OPTIMIZER" : "Adam",
    "LR" : EXPLORE_DURING_TUNING,
    "BATCH_SIZE" : 64,
    "N_EPOCHS" : [ 1000, 2500, 5000, 7500, 10000 ],
    "EARLY_STOPPING" : False,
    "DELTA": 0.05,
    "PATIENCE" : 20,
    "STRIDE" : 30,
    "N_MC_SAMPLES" : 1,                     # ~~~ relevant for droupout
    #
    # ~~~ For visualization
    "MAKE_GIF" : False,
    "HOW_OFTEN" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ for how many frames should the state after training be rendered
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ how many posterior predictive samples to plot
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ for dropout, if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "N_POSTERIOR_SAMPLES" : 100,            # ~~~ for dropout, how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "N_POSTERIOR_SAMPLES_EVALUATION" : 1000,
    "SHOW_DIAGNOSTICS" : False,
    "SHOW_PLOT" : False
}
#
# ~~~ Values that we want to test, for each one EXPLORE_DURING_TUNING
LR = np.linspace( 1e-5, 1e-2, 15 )
ARCHITECTURE = [
        "univar_NN",                    # ~~~ 2 hidden layers, 100 neurons each
        "univar_NN.univar_NN_300_300",  # ~~~ 2 hidden layers, 300 neurons each
        "univar_NN.univar_NN_500_500",  # ~~~ 2 hidden layers, 500 neurons each
        "univar_NN.univar_NN_750_750",  # ~~~ 2 hidden layers, 750 neurons each
        "univar_NN.univar_NN_1000_1000" # ~~~ 2 hidden layers, 1000 neurons each
    ]
DATA = [    # ~~~ two different train/val splits of the same data
        "univar_missing_middle_normalized_12",
        "univar_missing_middle_normalized_12_cross_fold",
    ]
#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir( os.path.join( folder_name, "experimental_models" ))
#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
for lr in LR:
    for architecture in ARCHITECTURE:
        for data in DATA:
            #
            # ~~~ Specify the specific values of the hyperparameters EXPLORE_DURING_TUNING
            hyperparameter_template["LR"] = lr
            hyperparameter_template["DATA"] = data
            hyperparameter_template["MODEL"] = architecture
            #
            # ~~~ Save the hyperparameters to a .json file
            tag = f"RUN_THIS_{count}.json"
            json_filename = os.path.join(folder_name,tag)
            dict_to_json( hyperparameter_template, json_filename, verbose=False )
            count += 1
print("")
print(f"Successfully created and populted the folder {folder_name} with {count-1} .json files. To run an hour of hyperparameter search, navigate to the directory of `tuning_loop.py` and say:")
print("")
print(f"`python tuning_loop.py --folder_name {folder_name} --hours 1`")
