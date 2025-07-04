import os
import torch
from bnns.utils.handling import dict_to_json
from bnns.experiments.paper.univar.dropout import (
    folder_name,
    DATA,
    ARCHITECTURE,
    LR,
    DROPOUT,
)


### ~~~
## ~~~ Create a folder and populate it with a whole bunch of JSON files
### ~~~

#
# ~~~ Define all hyperparmeters *including* even the ones not to be tuned
EXPLORE_DURING_TUNING = "placeholder value"
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "float",
    "SEED": 2024,
    #
    # ~~~ Which problem
    "DATA": EXPLORE_DURING_TUNING,
    "ARCHITECTURE": EXPLORE_DURING_TUNING,
    #
    # ~~~ For training
    "OPTIMIZER": "Adam",
    "LR": EXPLORE_DURING_TUNING,
    "BATCH_SIZE": 64,
    "N_EPOCHS": [10000, 20000, 30000],
    "EARLY_STOPPING": True,
    "DELTA": [-0.1, 0.1],
    "PATIENCE": [20, 50],
    "STRIDE": 15,
    "N_MC_SAMPLES": 1,  # ~~~ relevant for droupout
    "DROPOUT": EXPLORE_DURING_TUNING,  # ~~~ add dropout layers with p=DROPOUT to the model before training, if it is a purely deterministic model
    #
    # ~~~ For visualization
    "MAKE_GIF": False,
    "HOW_OFTEN": 50,  # ~~~ how many iterations we let pass before checking the validation data again
    "INITIAL_FRAME_REPETITIONS": 24,  # ~~~ N/A, because we aren't making a .gif
    "FINAL_FRAME_REPETITIONS": 48,  # ~~~ N/A, because we aren't making a .gif
    "HOW_MANY_INDIVIDUAL_PREDICTIONS": 6,  # ~~~ N/A, because we aren't graphing the model
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES": True,  # ~~~ N/A, because we aren't graphing the model
    "N_POSTERIOR_SAMPLES": 100,  # ~~~ how many samples to use to make the empirical distributions for computing metrics
    #
    # ~~~ For metrics and visualization
    "N_POSTERIOR_SAMPLES_EVALUATION": 1000,
    "SHOW_DIAGNOSTICS": False,
    "SHOW_PLOT": False,
}

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir(os.path.join(folder_name, "experimental_models"))

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
for lr in LR:
    for architecture in ARCHITECTURE:
        for data in DATA:
            for p in DROPOUT:
                #
                # ~~~ Specify the specific values of the hyperparameters EXPLORE_DURING_TUNING
                hyperparameter_template["LR"] = lr
                hyperparameter_template["DATA"] = data
                hyperparameter_template["ARCHITECTURE"] = architecture
                hyperparameter_template["DROPOUT"] = p
                #
                # ~~~ Save the hyperparameters to a .json file
                tag = f"RUN_THIS_{count}.json"
                json_filename = os.path.join(folder_name, tag)
                dict_to_json(hyperparameter_template, json_filename, verbose=False)
                count += 1

print("")
print(
    f"Successfully created and populted the folder {folder_name} with {count-1} .json files. To run an hour of hyperparameter search, navigate to the directory of `tuning_loop.py` and say:"
)
print("")
print(f"`python tuning_loop.py --folder_name {folder_name} --hours 1`")
