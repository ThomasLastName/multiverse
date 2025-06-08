

import os
import random
import torch
from bnns.utils.handling import dict_to_json, my_warn
from bnns.experiments.paper.univar.bnns import folder_name, N_HYPERPAR_SAMPLES, SEED, DATA, ARCHITECTURE, VARIATIONAL_FAMILY, LR, WEIGHTING, LIKELIHOOD_STD, FUNCTIONAL, MEASUREMENT_SET_SAMPLER, N_MEAS, PRIOR_J, POST_J, PRIOR_ETA, POST_ETA, PRIOR_M, POST_M



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
    # ~~~ Which problem, broadly spaking
    "DATA" : EXPLORE_DURING_TUNING,
    "ARCHITECTURE" : EXPLORE_DURING_TUNING,
    "MODEL": None,
    "VARIATIONAL_FAMILY" : EXPLORE_DURING_TUNING,
    #
    # ~~~ Any prior-specific hyper-parameters
    # EXPLORE_DURING_TUNING
    #
    # ~~~ For training
    "GAUSSIAN_APPROXIMATION" : False,       # ~~~ N/A when testing weight priors only
    "APPPROXIMATE_GAUSSIAN_MEAN" : False,   # ~~~ N/A when testing weight priors only
    "FUNCTIONAL" : EXPLORE_DURING_TUNING,   # ~~~ whether or to do functional training or (if False) BBB
    "MEASUREMENT_SET_SAMPLER" : EXPLORE_DURING_TUNING,  # ~~~ used for functional training; load this function from the same file where data is loaded from
    "N_MEAS" : EXPLORE_DURING_TUNING,           # ~~~ used for functional training; desired size of measurement set
    "EXACT_WEIGHT_KL" : False,              # ~~~ not found to result in meaningfully different results
    "PROJECTION_METHOD" : "HARD",           # ~~~ if "HARD", use projected gradient descent; else use the weird thing from the paper
    "PRIOR_J"   : EXPLORE_DURING_TUNING,    # ~~~ `J` in the SSGE of the prior score
    "POST_J"    : EXPLORE_DURING_TUNING,    # ~~~ `J` in the SSGE of the posterior score
    "PRIOR_ETA" : EXPLORE_DURING_TUNING,    # ~~~ `eta` in the SSGE of the prior score
    "POST_ETA"  : EXPLORE_DURING_TUNING,    # ~~~ `eta` in the SSGE of the posterior score
    "PRIOR_M"   : EXPLORE_DURING_TUNING,    # ~~~ `M` in the SSGE of the prior score
    "POST_M"    : EXPLORE_DURING_TUNING,    # ~~~ `M` in the SSGE of the posterior score
    "POST_GP_ETA" : 0.01,                   # ~~~ N/A for weight priors
    "LIKELIHOOD_STD" : EXPLORE_DURING_TUNING,
    "OPTIMIZER" : "Adam",
    "LR" : EXPLORE_DURING_TUNING,
    "BATCH_SIZE" : 600000,
    "N_EPOCHS" : [10000,20000,30000],
    "EARLY_STOPPING" : True,
    "DELTA": [ 0.05, 0.15 ],
    "PATIENCE" : [ 25, 75 ],
    "STRIDE" : 15,
    "N_MC_SAMPLES" : 1,
    "WEIGHTING" : None,                 # ~~~ lossely speaking, this determines how the minibatch estimator is normalized
    "DEFAULT_INITIALIZATION" : None,    # ~~~ how to initialize the variational standard deviations
    #
    # ~~~ For visualization0
    "MAKE_GIF" : False,
    "HOW_OFTEN" : 50,                       # ~~~ how many iterations we let pass before checking the validation data again
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ N/A, because we aren't making a .gif
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ N/A, because we aren't making a .gif
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ N/A, because we aren't graphing the model
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ N/A, because we aren't graphing the model
    "N_POSTERIOR_SAMPLES" : 30,             # ~~~ how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "EXTRA_STD" : True,
    "N_POSTERIOR_SAMPLES_EVALUATION" : 200, # ~~~ for computing our model evaluation metrics, posterior distributions are approximated as empirical dist.'s of this many samples
    "SHOW_DIAGNOSTICS" : False,
    "SHOW_PLOT" : False
}

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)

#
# ~~~ We won't have time to tune everything, so we'll randomly explore the parts of hyper-parameter space believed to be less important
def randomly_sample_less_important_hyperparameters(hyperparameter_template):
    #
    # ~~~ Randomize the rest, for the sake of compute time
    hyperparameter_template["SEED"] = random.choice(SEED)
    hyperparameter_template["DATA"] = random.choice(DATA)
    hyperparameter_template["ARCHITECTURE"] = random.choice(ARCHITECTURE)
    hyperparameter_template["VARIATIONAL_FAMILY"] = random.choice(VARIATIONAL_FAMILY)
    hyperparameter_template["LR"] = random.choice(LR)
    hyperparameter_template["WEIGHTING"] = random.choice(WEIGHTING)
    #
    # ~~~ Randomly set the hyperparameters of functional training
    n_meas = random.choice(N_MEAS)
    hyperparameter_template["N_MEAS"] = n_meas
    hyperparameter_template["MEASUREMENT_SET_SAMPLER"] = random.choice(MEASUREMENT_SET_SAMPLER) if n_meas>60 else "data_only"
    hyperparameter_template["PRIOR_J"] = random.choice(PRIOR_J)
    hyperparameter_template["POST_J"] = random.choice(POST_J)
    hyperparameter_template["PRIOR_ETA"] = random.choice(PRIOR_ETA)
    hyperparameter_template["POST_ETA"] = random.choice(POST_ETA)
    hyperparameter_template["PRIOR_M"] = random.choice(PRIOR_M)
    hyperparameter_template["POST_M"] = random.choice(POST_M)
    return hyperparameter_template

#
# ~~~ Loop over the hyperparameter grid, saving each one to a .json file called `RUN_THIS_<count>.json`
count = 1
random.seed(2025)
for likelihood_std in LIKELIHOOD_STD:
    for _ in range(N_HYPERPAR_SAMPLES):
        hyperparameter_template = randomly_sample_less_important_hyperparameters(hyperparameter_template)
        for functional in FUNCTIONAL:
            hyperparameter_template["LIKELIHOOD_STD"] = likelihood_std
            hyperparameter_template["FUNCTIONAL"] = functional
            #
            # ~~~ Save the hyperparameters to a .json file
            tag = f"RUN_THIS_{count}.json"
            json_filename = os.path.join(folder_name,tag)
            count += 1
            dict_to_json( hyperparameter_template, json_filename, verbose=False )

