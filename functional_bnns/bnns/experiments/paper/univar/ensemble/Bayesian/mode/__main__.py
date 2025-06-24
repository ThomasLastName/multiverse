import os
from glob import glob
import random
from bnns.utils.handling import dict_to_json, json_to_dict
from bnns.experiments.paper.univar.ensemble.Bayesian import folder_name as master_folder
from bnns.experiments.paper.univar.ensemble.Bayesian.mode import folder_name

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir(os.path.join(folder_name, "experimental_models"))

#
# ~~~
list_of_json_files = glob(os.path.join(master_folder, "*.json"))
if len(list_of_json_files) == 0:
    raise OSError(
        f"Unable to Locate any .json files in directory {master_folder}. This is likely because `__main__.py` was not run in that folder."
    )

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
random.seed(2025)
for hyperparameter_template in list_of_json_files:
    hyperparameter_template = json_to_dict(hyperparameter_template)
    hyperparameter_template["STEIN"] = False
    #
    # ~~~ Save the hyperparameters to a .json file
    tag = f"RUN_THIS_{count}.json"
    json_filename = os.path.join(folder_name, tag)
    count += 1
    dict_to_json(hyperparameter_template, json_filename, verbose=False)

print("")
print(
    f"Successfully created and populted the folder {folder_name} with {count-1} .json files. To run an hour of hyperparameter search, navigate to the directory of `tuning_loop.py` and say:"
)
print("")
print(f"`python tuning_loop.py --folder_name {folder_name} --hours 1`")
