

import os
import random
import torch
from quality_of_life.my_base_utils import dict_to_json, my_warn
from bnns.experiments.paper.univar.weight_training_vs_functional_training import folder_name, ARCHITECTURE, LR, MODEL, PI, SIGMA1, SIGMA2, PRIOR_TYPE, SCALE, LR, LIKELIHOOD_STD, FUNCTIONAL, PROJECTION_METHOD, DEFAULT_INITIALIZATION, MEASUREMENT_SET_SAMPLER, N_MEAS, PRIOR_J, POST_J, PRIOR_ETA, POST_ETA, PRIOR_M, POST_M



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
    "SEED" : 2025,
    #
    # ~~~ Which problem, broadly spaking
    "DATA" : "univar_missing_middle_normalized_12",
	"ARCHITECTURE" : EXPLORE_DURING_TUNING,
	"MODEL": EXPLORE_DURING_TUNING,
    #
    # ~~~ Any prior-specific hyper-parameters
    # EXPLORE_DURING_TUNING
    #
    # ~~~ For training
    "GAUSSIAN_APPROXIMATION" : False,       # ~~~ N/A since we are testing weight priors only in this round of experiments
    "APPPROXIMATE_GAUSSIAN_MEAN" : None,    # ~~~ N/A since we are testing weight priors only in this round of experiments
    "FUNCTIONAL" : EXPLORE_DURING_TUNING,   # ~~~ whether or to do functional training or (if False) BBB
    "MEASUREMENT_SET_SAMPLER" : EXPLORE_DURING_TUNING,  # ~~~ used for functional training; load this function from the same file where data is loaded from
    "N_MEAS" : EXPLORE_DURING_TUNING,           # ~~~ used for functional training; desired size of measurement set
    "EXACT_WEIGHT_KL" : IT_DEPENDS,             # ~~~ use when possible
    "PROJECTION_METHOD" : EXPLORE_DURING_TUNING,# ~~~ if "HARD", use projected gradient descent; else use the weird thing from the paper
    "PRIOR_J"   : EXPLORE_DURING_TUNING,    # ~~~ `J` in the SSGE of the prior score
    "POST_J"    : EXPLORE_DURING_TUNING,    # ~~~ `J` in the SSGE of the posterior score
    "PRIOR_ETA" : EXPLORE_DURING_TUNING,    # ~~~ `eta` in the SSGE of the prior score
    "POST_ETA"  : EXPLORE_DURING_TUNING,    # ~~~ `eta` in the SSGE of the posterior score
    "PRIOR_M"   : EXPLORE_DURING_TUNING,    # ~~~ `M` in the SSGE of the prior score
    "POST_M"    : EXPLORE_DURING_TUNING,    # ~~~ `M` in the SSGE of the posterior score
    "POST_GP_ETA" : None,                   # ~~~ N/A for weight priors
    "LIKELIHOOD_STD" : EXPLORE_DURING_TUNING,
    "OPTIMIZER" : "Adam",
    "LR" : EXPLORE_DURING_TUNING,
    "BATCH_SIZE" : 600,
    "N_EPOCHS" : [10000,20000,30000],
    "EARLY_STOPPING" : True,
    "DELTA": [ -0.1, 0.1 ],
    "PATIENCE" : [ 25, 50 ],
    "STRIDE" : 15,
    "N_MC_SAMPLES" : 1,
    "WEIGHTING" : "standard",   # ~~~ lossely speaking, this determines how the minibatch estimator is normalized
    "DEFAULT_INITIALIZATION" : EXPLORE_DURING_TUNING,   # ~~~ how to initialize the variational standard deviations
    #
    # ~~~ For visualization0
    "MAKE_GIF" : False,
    "HOW_OFTEN" : 50,                       # ~~~ how many iterations we let pass before checking the validation data again
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ N/A, because we aren't making a .gif
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ N/A, because we aren't making a .gif
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ N/A, because we aren't graphing the model
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ N/A, because we aren't graphing the model
    "N_POSTERIOR_SAMPLES" : 50,             # ~~~ how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "EXTRA_STD" : False,
    "N_POSTERIOR_SAMPLES_EVALUATION" : 100, # ~~~ for computing our model evaluation metrics, posterior distributions are approximated as empirical dist.'s of this many samples
    "SHOW_DIAGNOSTICS" : False,
    "SHOW_PLOT" : False
}

#
# ~~~ Create and populate a folder for the hyperparameter search
os.mkdir(folder_name)
os.mkdir( os.path.join( folder_name, "experimental_models" ))

#
# ~~~ We won't have time to tune everything, so we'll randomly explore the parts of hyper-parameter space believed to be less important
def randomly_sample_less_important_hyperparameters(hyperparameter_template):
    #
    # ~~~ Randomize the rest, for the sake of compute time
    projection_method = random.choice(PROJECTION_METHOD)
    hyperparameter_template["PROJECTION_METHOD"] = projection_method
    hyperparameter_template["DEFAULT_INITIALIZATION"] = random.choice( DEFAULT_INITIALIZATION if projection_method=="HARD" else DEFAULT_INITIALIZATION[1:] )
    #
    # ~~~ Randomly set the prior hyperparameters
    if model=="GaussianBNN":
        hyperparameter_template["prior_type"] = random.choice(PRIOR_TYPE)
        hyperparameter_template["scale"] = random.choice(SCALE)
    else:
        hyperparameter_template["pi"] = random.choice(PI)
        hyperparameter_template["sigma1"] = random.choice(SIGMA1)
        hyperparameter_template["sigma2"] = random.choice(SIGMA2)
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
for lr in LR:
	for architecture in ARCHITECTURE:
		for model in MODEL: # ~~~ i.e., prior
			for likelihood_std in LIKELIHOOD_STD:
				hyperparameter_template = randomly_sample_less_important_hyperparameters(hyperparameter_template)
				for functional in FUNCTIONAL:
					hyperparameter_template["LR"] = lr
					hyperparameter_template["ARCHITECTURE"] = architecture
					hyperparameter_template["MODEL"] = model
					hyperparameter_template["LIKELIHOOD_STD"] = likelihood_std
					hyperparameter_template["FUNCTIONAL"] = functional
					hyperparameter_template["EXACT_WEIGHT_KL"] = (MODEL=="GaussianBNN") and (not functional)
                    #
					# ~~~ Some post-hoc tweaks to avoid running out of RAM
					if ARCHITECTURE == "univar_NN.univar_NN_500_500_500_500":
						if hyperparameter_template["PRIOR_M"] == 2500:
							hyperparameter_template["PRIOR_M"] = 2000
                    #
                    # ~~~ Save the hyperparameters to a .json file
					tag = f"RUN_THIS_{count}.json"
					json_filename = os.path.join(folder_name,tag)
					count += 1
					dict_to_json( hyperparameter_template, json_filename, verbose=False )

print("")
print(f"Successfully created and populted the folder {folder_name} with {count-1} .json files. To run an hour of hyperparameter search, navigate to the directory of `tuning_loop.py` and say:")
print("")
print(f"`python tuning_loop.py --folder_name {folder_name} --hours 1`")

