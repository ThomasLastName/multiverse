import os
import random
import torch
from bnns.utils.handling import dict_to_json, my_warn, fdict
from bnns.experiments.paper.univar.weight_priors import (
    folder_name,
    ARCHITECTURE,
    VARIATIONAL_FAMILY,
    MODEL,
    PI,
    SIGMA1,
    SIGMA2,
    PRIOR_TYPE,
    SCALE,
    LR,
    LIKELIHOOD_STD,
    FUNCTIONAL,
    PROJECTION_METHOD,
    DEFAULT_INITIALIZATION,
    MEASUREMENT_SET_SAMPLER,
    N_MEAS,
    PRIOR_J,
    POST_J,
    PRIOR_ETA,
    POST_ETA,
    PRIOR_M,
    POST_M,
)


### ~~~
## ~~~ Create a folder and populate it with a whole bunch of JSON files
### ~~~

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties("cuda")
    GPU_RAM = props.total_memory / (1024**3)  # Convert bytes to gigabytes
    print("")
    print(
        f'    Experiments will be run on device "cuda" which has {GPU_RAM:.2f} GB of RAM'
    )
    if GPU_RAM < 7.5:
        my_warn(
            "These experiments have been run on a laptop with an 8GB NVIDIA 4070. They have not been tested on a GPU with less than 8GB of ram; it is possible that a cuda 'out of memory' error could arise."
        )

#
# ~~~ Define all hyperparameters *including* even the ones not to be tuned
EXPLORE_DURING_TUNING = "placeholder value"
IT_DEPENDS = "placeholder value"
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "float",
    "SEED": 2025,
    #
    # ~~~ The data, the prior distribution, and the variational family
    "DATA": "univar_missing_middle_normalized_12",
    "ARCHITECTURE": EXPLORE_DURING_TUNING,
    "MODEL": EXPLORE_DURING_TUNING,
    "VARIATIONAL_FAMILY": EXPLORE_DURING_TUNING,
    "PRIOR_HYPERPARAMETERS": IT_DEPENDS,
    #
    # ~~~ Training hyper-parameters for neural networks, in general
    "OPTIMIZER": "Adam",
    "LR": EXPLORE_DURING_TUNING,
    "BATCH_SIZE": 600,
    "N_EPOCHS": [10000, 20000, 30000],
    "EARLY_STOPPING": True,
    "DELTA": [0.05, 0.15],
    "PATIENCE": [25, 75],
    "STRIDE": 15,
    "HOW_OFTEN": 50,
    #
    # ~~~ Training hyper-parameters introduced by BNNs, in general
    "FUNCTIONAL": EXPLORE_DURING_TUNING,
    "PROJECTION_METHOD": EXPLORE_DURING_TUNING,
    "LIKELIHOOD_STD": EXPLORE_DURING_TUNING,
    "N_MC_SAMPLES": 1,
    "WEIGHTING": "standard",
    "DEFAULT_INITIALIZATION": EXPLORE_DURING_TUNING,
    "EXTRA_STD": True,
    "N_POSTERIOR_SAMPLES_EVALUATION": 100,
    "N_POSTERIOR_SAMPLES": 50,
    "SHOW_DIAGNOSTICS": False,
    #
    # Training hyper-parameters that are specific to BBB
    "EXACT_WEIGHT_KL": False,
    #
    # Training hyper-parameters that are specific to fBNNs
    "N_MEAS": EXPLORE_DURING_TUNING,
    "MEASUREMENT_SET_SAMPLER": EXPLORE_DURING_TUNING,
    "GAUSSIAN_APPROXIMATION": False,
    # 
    # ~~~ Training hyper-parameters introduced by the SSGE (only used when FUNCTIONAL is true but GAUSSIAN_APPROXIMATION is false)
    "SSGE_HYPERPARAMETERS": {
        "prior_M": EXPLORE_DURING_TUNING,
        "post_M": EXPLORE_DURING_TUNING,
        "prior_eta": EXPLORE_DURING_TUNING,
        "post_eta": EXPLORE_DURING_TUNING,
        "prior_J": EXPLORE_DURING_TUNING,
        "post_J": EXPLORE_DURING_TUNING,
    },
    #
    # ~~~ Training hyper-parameters introduced by Rudner et al. 2023 (only used when FUNCTIONAL and GUASSIAN_APPROXIMATION are true and MODEL is "GPPrior2023BNN")
    "APPROXIMATE_GAUSSIAN_MEAN": False,
    "POST_GP_ETA": None,    # ~~~ only applicable for GP priors whereas we are testing weight priors in this round of experiments
    #
    # ~~~ Visualization options (only when there's a univariate input and univariate output)
    "MAKE_GIF": False,
    "TITLE": None,
    "SHOW_PLOT": False,
    "INITIAL_FRAME_REPETITIONS": None,
    "FINAL_FRAME_REPETITIONS": None,
    "HOW_MANY_INDIVIDUAL_PREDICTIONS": None,
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES": True,
}


### ~~~
## ~~~ Randomized less important params
### ~~~

def randomly_sample_less_important_hyperparameters(config):
    #
    # ~~~ Misc.
    config["LR"] = random.choice(LR)
    projection_method = random.choice(PROJECTION_METHOD)
    config["PROJECTION_METHOD"] = projection_method
    config["DEFAULT_INITIALIZATION"] = random.choice(
        DEFAULT_INITIALIZATION if projection_method.upper() == "HARD" else DEFAULT_INITIALIZATION[1:]
    )
    config["VARIATIONAL_FAMILY"] = random.choice(VARIATIONAL_FAMILY)
    #
    # ~~~ Prior hyperparameters
    if config["MODEL"] == "MixturePrior2015BNN":
        config["PRIOR_HYPERPARAMETERS"] = {
            "pi": random.choice(PI),
            "sigma1": random.choice(SIGMA1),
            "sigma2": random.choice(SIGMA2),
        }
    elif config["MODEL"] == "IndepLocScalePriorBNN":
        config["PRIOR_HYPERPARAMETERS"] = {
            "prior_type": random.choice(PRIOR_TYPE),
            "scale": random.choice(SCALE),
        }
    else:
        raise ValueError(f"Unrecognized model: '{config['MODEL']}' in\n{fdict(config)}\n")
    #
    # ~~~ fBNN hyperparameters
    n_meas = random.choice(N_MEAS)
    config["N_MEAS"] = n_meas
    config["MEASUREMENT_SET_SAMPLER"] = (
        random.choice(MEASUREMENT_SET_SAMPLER) if n_meas > 60 else "data_only"
    )
    #
    # ~~~ SSGE hyperparameters
    config["SSGE_HYPERPARAMETERS"] = {
        "prior_J": random.choice(PRIOR_J),
        "post_J": random.choice(POST_J),
        "prior_eta": random.choice(PRIOR_ETA),
        "post_eta": random.choice(POST_ETA),
        "prior_M": random.choice(PRIOR_M),
        "post_M": random.choice(POST_M),
    }
    return config


### ~~~
## ~~~ Loop over hyperparameter grid
### ~~~

# fmt: off
os.mkdir(folder_name)
os.mkdir(os.path.join(folder_name, "experimental_models"))
count = 1
random.seed(2025)
for architecture in ARCHITECTURE:
    for model in MODEL:  # ~~~ i.e., prior
        for likelihood_std in LIKELIHOOD_STD:
            hyperparameter_template["ARCHITECTURE"] = architecture
            hyperparameter_template["MODEL"] = model
            hyperparameter_template["LIKELIHOOD_STD"] = likelihood_std
            hyperparameter_template = randomly_sample_less_important_hyperparameters(hyperparameter_template)
            for functional in FUNCTIONAL:
                hyperparameter_template["FUNCTIONAL"] = functional
                #
                # ~~~ Save the hyperparameters to a .json file
                tag = f"RUN_THIS_{count}.json"
                json_filename = os.path.join(folder_name, tag)
                count += 1
                dict_to_json(hyperparameter_template, json_filename, verbose=False)

print("")
print(f"Successfully created and populted the folder {folder_name} with {count-1} .json files. To run an hour of hyperparameter search, navigate to the directory of `tuning_loop.py` and say:")
print("")
print(f"`python tuning_loop.py --folder_name {folder_name} --hours 1`")
