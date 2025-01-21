

import os
import torch
import numpy as np
from quality_of_life.my_base_utils import dict_to_json, my_warn
from bnns.experiments.paper.univar.BBB import folder_name



### ~~~
## ~~~ Create a folder and populate it with a whole bunch of JSON files
### ~~~

if torch.cuda.is_available():
	props = torch.cuda.get_device_properties("cuda")
	GPU_RAM = props.total_memory / (1024 ** 3)  # Convert bytes to gigabytes
	print("")
	print(f'    Experiments will be run on device "cuda" which has {GPU_RAM:.2f} GB of RAM')
	if GPU_RAM < 7.5:
		my_warn("These experiments have been run on a laptop with an 8GB NVIDIA 4070. They have not been tested on a GPU with less than 8GB of ram; it is possible that a cuda 'out of memory' error could arise")

#
# ~~~ Define all hyperparmeters *including* even the ones not to be tuned
EXPLORE_DURING_TUNING = "placeholder value"
IT_DEPENDS = "placeholder value"
hyperparameter_template = {
	"DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
	"DTYPE": "float",
	"SEED": 2024,
	"DATA": "univar_missing_middle_normalized_12",
	"MODEL": EXPLORE_DURING_TUNING,
	"GAUSSIAN_APPROXIMATION": False,
	"APPPROXIMATE_GAUSSIAN_MEAN" : False,
	"FUNCTIONAL": False,
	"EXACT_WEIGHT_KL" : True,
	"PROJECT": EXPLORE_DURING_TUNING,
	"PROJECTION_TOL": 1e-06,
	"PRIOR_J": None,
	"POST_J": None,
	"PRIOR_eta": None,
	"POST_eta": None,
	"PRIOR_M": None,
	"POST_M": None,
	"POST_GP_eta": None,
	"CONDITIONAL_STD": EXPLORE_DURING_TUNING,
	"OPTIMIZER": "Adam",
	"LR": EXPLORE_DURING_TUNING,
	"BATCH_SIZE": 64,
	"N_EPOCHS": [ 20000, 50000, 100000 ],
	"EARLY_STOPPING": True,
	"DELTA": IT_DEPENDS,
	"PATIENCE": IT_DEPENDS,
	"STRIDE": 20,
	"N_MC_SAMPLES": 1,
	"WEIGHTING": EXPLORE_DURING_TUNING,
	"INITIALIZE_AT_PRIOR" : True,
	"MAKE_GIF": False,
	"HOW_OFTEN": 20,
	"INITIAL_FRAME_REPETITIONS": None,
	"FINAL_FRAME_REPETITIONS": None,
	"HOW_MANY_INDIVIDUAL_PREDICTIONS": 0,
	"VISUALIZE_DISTRIBUTION_USING_QUANTILES": None,
	"N_POSTERIOR_SAMPLES": 100,
	"EXTRA_STD": True,
	"N_POSTERIOR_SAMPLES_EVALUATION": 1000,
	"SHOW_DIAGNOSTICS" : False,
	"SHOW_PLOT" : False
}

#
# ~~~ Values that we want to test, for each one EXPLORE_DURING_TUNING
LR = [ 0.001, 0.0005, 0.0001 ]
ARCHITECTURE = [
        "univar_BNN.univar_BNN_3000",   # ~~~ shallow and wide: 1 hidden layer with 3000 neurons
        "univar_BNN.univar_BNN_300_300",    # ~~~ 2 hidden layers, 300 neurons each
        "univar_BNN.univar_BNN_1000_1000", 	# ~~~ 2 hidden layers, 1000 neurons each
        "univar_BNN.univar_BNN_300_300_300_300",    # ~~~ 4 hidden layers, 300 neurons each
        "univar_BNN.univar_BNN_1000_1000_1000_1000" # ~~~ 4 hidden layers, 1000 neurons each
    ]
CONDITIONAL_STD = np.concatenate([
        np.geomspace( 1e-4, 1e-2, 7 ),  # ~~~ a grid
        [0.0156]                        # ~~~ the theoretically correct value (bigger than the max of the grid)
    ])

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir( os.path.join( folder_name, "experimental_models" ))

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
for lr in LR:
	for architecture in ARCHITECTURE:
		for conditional_std in CONDITIONAL_STD:
			for projected_gradient_descent in (True,False):
				for weighting in ("standard","naive"):
					#
					# ~~~ Specify the specific values of the hyperparameters EXPLORE_DURING_TUNING
					hyperparameter_template["LR"] = lr
					hyperparameter_template["MODEL"] = architecture
					hyperparameter_template["CONDITIONAL_STD"] = conditional_std
					hyperparameter_template["PROJECT"] 	= projected_gradient_descent
					hyperparameter_template["PATIENCE"] = 1000 if projected_gradient_descent else 200
					hyperparameter_template["DELTA"] 	= -0.1 if projected_gradient_descent else -0.5
					hyperparameter_template["WEIGHTING"] = weighting
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

