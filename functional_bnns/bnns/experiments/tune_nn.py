
import os
import argparse
from subprocess import run
import numpy as np
from bnns.utils import generate_json_filename
from quality_of_life.my_base_utils import dict_to_json, my_warn



### ~~~
## ~~~ Define the hyperparameters and problem structure
### ~~~

#
# ~~~ Define all hyperparmeters *including* even the ones not to be tuned
TO_BE_TUNED = "placeholder_value"
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE" : "cpu",
    "dtype" : "float",
    "seed" : 2024,
    #
    # ~~~ Which problem
    "data" : "univar_missing_middle_normalized_12",
    "model" : TO_BE_TUNED,
    #
    # ~~~ For training
    "Optimizer" : "Adam",
    "lr" : TO_BE_TUNED,
    "batch_size" : TO_BE_TUNED,
    "n_epochs" : TO_BE_TUNED,
    #
    # ~~~ For metrics and visualization
    "show_diagnostics" : False
}

#
# ~~~ Values that we want to test for each one TO_BE_TUNED
LR = np.linspace( 1e-5, 1e-2, 25 )
N_EPOCHS = np.linspace( 500, 20000, 25 )
BATCH_SIZE = [ 10, 20, 50, 100 ]
ARCHITECTURE = [
        [100,100],
        [300,300],
        [500,500],
        [750,750],
        [1000,1000],
        [100,100,100],
        [300,300,300],
        [500,500,500],
        [750,750,750],
        [1000,1000,1000],
    ]



### ~~~
## ~~~ Search over the hyperparameter grid with the defined problem structure
### ~~~ 

#
# ~~~ Gather metadata
parser = argparse.ArgumentParser()
try:
    parser.add_argument( '--folder_name', type=str, required=True )
except:
    print("")
    print("    Hint: try `python tune_nn.py --json ?????`")
    print("")
    raise
parser.add_argument( '--save_trained_models', action=argparse.BooleanOptionalAction )
args = parser.parse_args()
folder_name = args.folder_name
save_trained_models = args.save_trained_models

#
# ~~~ Create the folder `folder_name` as a subdirectory of `bnns.experiments`
try:
    os.mkdir(folder_name)
except FileExistsError:
    folder_is_empty = len(os.listdir(folder_name))==0
    if not folder_is_empty:
        my_warn(f"Folder {folder_name} already exists. The .json files from this experiement will be added to a non-empty folder.")

#
# ~~~ Create the folder `folder_name/experimental_models`
if save_trained_models:
    try:
        os.mkdir(os.path.join(folder_name,"experimental_models"))
    except FileExistsError:
        folder_is_empty = len(os.listdir(folder_name))==0
        if not folder_is_empty:
            my_warn(f"Folder `{folder_name}/experimental_models` already exists. The .json files from this experiement will be added to a non-empty folder.")

#
# ~~~ Loop over the hyperparameter grid
for lr in LR:
    for n_epochs in N_EPOCHS:
        for batch_size in BATCH_SIZE:
            for architecture in ARCHITECTURE:
                #
                # ~~~ Specify the specific values of the hyperparameters TO_BE_TUNED
                hyperparameter_template["lr"] = lr
                hyperparameter_template["n_epochs"] = n_epochs
                hyperparameter_template["batch_size" ] = batch_size
                hyperparameter_template["model"] = architecture
                #
                # ~~~ Save the hyperparameters to a .json file
                tag = generate_json_filename()
                json_filename = os.path.join(folder_name,tag)
                dict_to_json( hyperparameter_template, json_filename )
                #
                # ~~~ Run the training script on that dictionary of hyperparameters
                basic_command = f"python train_nn.py --json {json_filename} --overwrite_json"
                if save_trained_models:
                    basic_command += f" --model_save_dir {os.path.join(folder_name,'experimental_models',tag)}"
                run( basic_command, shell=True )
