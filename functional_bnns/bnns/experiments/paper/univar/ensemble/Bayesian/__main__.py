

import os
from glob import glob
import random
from bnns.utils.handling import dict_to_json, json_to_dict
from bnns.experiments.paper.univar.ensemble import folder_name as parent_folder
from bnns.experiments.paper.univar.ensemble.Bayesian import folder_name, LIKELIHOOD_STD, WEIGHTING

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir(os.path.join( folder_name, "experimental_models" ))

#
# ~~~
list_of_json_files = glob(os.path.join(parent_folder,'*.json'))
if len(list_of_json_files)==0: raise OSError(f"Unable to Locate any .json files in directory {parent_folder}. This is likely because `__main__.py` was not run in that folder.")

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
random.seed(2025)
for hyperparameter_template in list_of_json_files:
    hyperparameter_template = json_to_dict(hyperparameter_template)
    hyperparameter_template["BAYESIAN"] = True
    hyperparameter_template["EXTRA_STD"] = True
    hyperparameter_template["WEIGHTING"] = random.choice(WEIGHTING)
    for likelihood_std in LIKELIHOOD_STD:
        hyperparameter_template["LIKELIHOOD_STD"] = likelihood_std
        #
        # ~~~ Save the hyperparameters to a .json file
        tag = f"RUN_THIS_{count}.json"
        json_filename = os.path.join(folder_name,tag)
        count += 1
        dict_to_json( hyperparameter_template, json_filename, verbose=False )
