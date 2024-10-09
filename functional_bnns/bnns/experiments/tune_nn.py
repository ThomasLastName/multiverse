
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
TO_BE_TUNED = "placeholder value"
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE" : "cuda",
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
    "n_MC_samples" : 1,                     # ~~~ relevant for droupout
    #
    # ~~~ For visualization
    "make_gif" : False,
    "how_often" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "initial_frame_repetitions" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "final_frame_repetitions" : 48,         # ~~~ for how many frames should the state after training be rendered
    "how_many_individual_predictions" : 6,  # ~~~ how many posterior predictive samples to plot
    "visualize_bnn_using_quantiles" : True, # ~~~ for dropout, if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "n_posterior_samples" : 100,            # ~~~ for dropout, how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "n_posterior_samples_evaluation" : 1000,
    "show_diagnostics" : False
}


#
# ~~~ Values that we want to test for each one TO_BE_TUNED
LR = np.linspace( 1e-5, 1e-2, 25 )
N_EPOCHS = [ int(obj) for obj in np.linspace( 500, 20000, 25 ) ]
BATCH_SIZE = [ 10, 20, 50, 100 ]
ARCHITECTURE = [
        "univar_NN",            # ~~~ 2 hidden layers, 100 neurons each
        "univar_NN_300_300",    # ~~~ 2 hidden layers, 300 neurons each
        "univar_NN_500_500",    # ~~~ 2 hidden layers, 500 neurons each
        "univar_NN_750_750",    # ~~~ 2 hidden layers, 750 neurons each
        "univar_NN_1000_1000"   # ~~~ 2 hidden layers, 1000 neurons each
    ]



### ~~~
## ~~~ Search over the hyperparameter grid with the defined problem structure
### ~~~ 

#
# ~~~ Gather metadata
parser = argparse.ArgumentParser()
parser.add_argument( '--folder_name', type=str, required=True )
parser.add_argument( '--save_trained_models', action=argparse.BooleanOptionalAction )
args = parser.parse_args()
folder_name = args.folder_name
save_trained_models = (args.save_trained_models is not None)

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
                dict_to_json( hyperparameter_template, json_filename, verbose=False )
                #
                # ~~~ Run the training script on that dictionary of hyperparameters
                basic_command = f"python train_nn.py --json {json_filename} --overwrite_json"
                if save_trained_models:
                    basic_command += f" --model_save_dir {os.path.join(folder_name,'experimental_models',tag)}"
                output = run( basic_command, shell=True )
                #
                # ~~~ Break out of the loop if there was an error in `train_nn.py`
                if not output.returncode==0:
                    break
