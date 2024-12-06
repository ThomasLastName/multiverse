
import os
import argparse
from glob import glob
from time import time
from subprocess import run
from bnns.utils import generate_json_filename
from quality_of_life.my_base_utils import dict_to_json, json_to_dict



### ~~~
## ~~~ Assume that the folder `folder_name` is already populated with the .json files for which we want to run `python train_<algorithm>.py --json file_from_folder.json`
### ~~~ 

#
# ~~~ Gather metadata
parser = argparse.ArgumentParser()
parser.add_argument( '--folder_name', type=str, required=True )
parser.add_argument( '--save_trained_models', action=argparse.BooleanOptionalAction )   # ~~~ default to True if unspecified
parser.add_argument( '--hours', type=float )    # ~~~ default to float("inf") if unspecified
args = parser.parse_args()
folder_name = args.folder_name
save_trained_models = True if (args.save_trained_models is None) else args.save_trained_models
hours = args.hours if (args.hours is not None) else float("inf")

#
# ~~~ Load all the json files in `folder_name` that start with "RUN_THIS"
list_of_json_filenames_in_folder = glob(os.path.join( folder_name, "*.json" ))
filenames_only = [ os.path.split(f)[1] for f in list_of_json_filenames_in_folder ]
sorted_list_of_filenames_starting_with_RUN_THIS = sorted(
        [ f for f in filenames_only if f.startswith("RUN_THIS") ],
        key = lambda x: int(x.split('_')[2].split('.')[0])
    )
N = int( sorted_list_of_filenames_starting_with_RUN_THIS[-1][len("RUN_THIS_"):].strip(".json") )    # ~~~ e.g., if sorted_list_of_filenames_starting_with_RUN_THIS[-1]=="RUN_THIS_199.json", then N==199

#
# ~~~ For `hours` hours, run the remaining experiments
start_time = time()
minutes_since_start_time = 0.
while (minutes_since_start_time < hours*60) and len(sorted_list_of_filenames_starting_with_RUN_THIS)>0:
    #
    # ~~~ Load the .json file sorted_list_of_filenames_starting_with_RUN_THIS[0]
    experiment_filename = sorted_list_of_filenames_starting_with_RUN_THIS.pop(0)
    count = int( experiment_filename[len("RUN_THIS_"):].strip(".json") )    # ~~~ e.g., if sorted_list_of_filenames_starting_with_RUN_THIS[-1]=="RUN_THIS_62.json", then count==62
    experiment_filename = os.path.join( folder_name, experiment_filename )
    hyperparameter_dict = json_to_dict(experiment_filename)
    #
    # ~~~ Create a new .json file (with a different name) to store the results
    print("")
    tag = generate_json_filename( message=f"EXPERIMENT {count}/{N}" )
    print("")
    result_filename = os.path.join( folder_name, tag )
    dict_to_json( hyperparameter_dict, result_filename, verbose=False )
    #
    # ~~~ Infer which training script to run, based on the hyperparameters
    if any( key=="GAUSSIAN_APPROXIMATION" for key in hyperparameter_dict.keys() ):
        algorithm = "bnn"
    elif any( key=="STEIN" for key in hyperparameter_dict.keys() ):
        algorithm = "ensemble"
    else:
        algorithm = "nn"
    #
    # ~~~ Run the training script on that dictionary of hyperparameters
    command = f"python train_{algorithm}.py --json {result_filename} --overwrite_json"
    if save_trained_models:
        command += f" --model_save_dir {os.path.join( folder_name, 'experimental_models' )}"
    output = run( command, shell=True )
    #
    # ~~~ Break out of the loop if there was an error in `train_nn.py`
    if not output.returncode==0:
        break
    #
    # ~~~ Delete the .json file that prescribed the experiment now run
    os.remove(experiment_filename)
    #
    # ~~~ Record how long we've been at it
    minutes_since_start_time = (time()-start_time)/60
