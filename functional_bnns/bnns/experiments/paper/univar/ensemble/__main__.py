
import os
import random
import torch
from bnns.utils.handling import dict_to_json, my_warn
from bnns.experiments.paper.univar.ensemble import folder_name, N_HYPERPAR_SAMPLES, SEED, DATA, ARCHITECTURE, LR



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
    #
    # ~~~ Misc.
    "DEVICE" : "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE" : "float",
    "SEED" : EXPLORE_DURING_TUNING,
    #
    # ~~~ Which problem
    "DATA" : EXPLORE_DURING_TUNING,
    "ARCHITECTURE" : EXPLORE_DURING_TUNING,
    #
    # ~~~ For training
    "STEIN" : EXPLORE_DURING_TUNING,
	"BAYESIAN" : EXPLORE_DURING_TUNING,
    "LIKELIHOOD_STD" : EXPLORE_DURING_TUNING,
    "BW" : None,
    "N_MODELS" : 30,
    "OPTIMIZER" : "Adam",
    "LR" : EXPLORE_DURING_TUNING,
    "BATCH_SIZE" : 64,
    "N_EPOCHS" : [10000,20000,30000],
    "EARLY_STOPPING" : True,
    "DELTA": [ 0.05, 0.15 ],
    "PATIENCE" : [ 25, 75 ],
    "STRIDE" : 15,
    "WEIGHTING" : EXPLORE_DURING_TUNING,    # ~~~ lossely speaking, this determines how the minibatch estimator is normalized
    #
    # ~~~ For visualization (only applicable on 1d data)
    "MAKE_GIF" : False,
    "TITLE" : "title of my gif",            # ~~~ if MAKE_GIF is True, this will be the file name of the created .gif
    "HOW_OFTEN" : 50,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ for how many frames should the state after training be rendered
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ how many posterior predictive samples to plot
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    #
    # ~~~ For metrics and visualization
    "EXTRA_STD" : IT_DEPENDS,
    "SHOW_DIAGNOSTICS" : False,
    "SHOW_PLOT" : False
}

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)

#
# ~~~ We won't have time to tune everything, so we'll randomly explore the parts of hyper-parameter space believed to be less important
def randomly_sample_less_important_hyperparameters(hyperparameter_template):
	hyperparameter_template["SEED"] = random.choice(SEED)
	hyperparameter_template["DATA"] = random.choice(DATA)
	hyperparameter_template["ARCHITECTURE"] = random.choice(ARCHITECTURE)
	hyperparameter_template["LR"] = random.choice(LR)
	return hyperparameter_template

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
random.seed(2025)
for _ in range(N_HYPERPAR_SAMPLES):
    hyperparameter_template = randomly_sample_less_important_hyperparameters(hyperparameter_template)
    #
    # ~~~ Save the hyperparameters to a .json file
    tag = f"RUN_THIS_{count}.json"
    json_filename = os.path.join(folder_name,tag)
    count += 1
    dict_to_json( hyperparameter_template, json_filename, verbose=False )
