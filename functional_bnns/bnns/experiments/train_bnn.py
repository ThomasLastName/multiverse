
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from statistics import mean as avg
from matplotlib import pyplot as plt
from importlib import import_module
from itertools import product
from time import time
import argparse
import sys
import os

#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, set_Dataset_attributes, generate_json_filename, convert_to_list_and_check_items, non_negative_list, EarlyStopper
from bnns.metrics import *

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_torch_utils         import convert_Dataset_to_Tensors
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, print_dict, my_warn, process_for_saving



### ~~~
## ~~~ Config/setup
### ~~~

#
# ~~~ Template for what the dictionary of hyperparmeters should look like
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE" : "cpu",
    "DTYPE" : "float",
    "SEED" : 2024,
    #
    # ~~~ Which problem
    "DATA" : "univar_missing_middle",
    "MODEL" : "univar_BNN",
    #
    # ~~~ For training
    "GAUSSIAN_APPROXIMATION" : True,    # ~~~ in an fBNN use a first order Gaussian approximation like Rudner et al.
    "APPPROXIMATE_GAUSSIAN_MEAN" : True,# ~~~ whether to compute exactly, or approximately, the mean from eq'n (14) in https://arxiv.org/pdf/2312.17199
    "FUNCTIONAL" : False,       # ~~~ whether or to do functional training or (if False) BBB
    "EXACT_WEIGHT_KL" : True,   # ~~~ whether to use the exact KL divergence between the prior and posterior (True) or a Monte-Carlo approximation (False)
    "PROJECT" : True,           # ~~~ if True, use projected gradient descent; else use the weird thing from the paper
    "PROJECTION_TOL" : 1e-6,    # ~~~ for numerical reasons, project onto [PROJECTION_TOL,Inf), rather than onto [0,Inft)
    "PRIOR_J"   : 100,          # ~~~ `J` in the SSGE of the prior score
    "POST_J"    : 10,           # ~~~ `J` in the SSGE of the posterior score
    "PRIOR_eta" : 0.5,          # ~~~ `eta` in the SSGE of the prior score
    "POST_eta"  : 0.5,          # ~~~ `eta` in the SSGE of the posterior score
    "PRIOR_M"   : 4000,         # ~~~ `M` in the SSGE of the prior score
    "POST_M"    : 40,           # ~~~ `M` in the SSGE of the posterior score
    "POST_GP_eta" : 0.001,      # ~~~ the level of the "stabilizing noise" added to the Gaussian approximation of the posterior distribution if `gaussian_approximation` is True
    "CONDITIONAL_STD" : 0.19,
    "OPTIMIZER" : "Adam",
    "LR" : 0.0005,
    "BATCH_SIZE" : 64,
    "N_EPOCHS" : 200,
    "EARLY_STOPPING" : True,
    "DELTA": 0.05,
    "PATIENCE" : 20,
    "STRIDE" : 30,
    "N_MC_SAMPLES" : 1,
    "WEIGHTING" : "standard",           # ~~~ lossely speaking, this determines how the minibatch estimator is normalized
    "DEFAULT_INITIALIZATION" : "new",   # ~~~ whether or not to take the prior as the initialization of the posterior
    "GP_PRIOR" : False,                 # ~~~ whether or not to use a Gaussian process prior
    "GP_PRIOR_ETA" : 0.001,             # ~~~ "stabilizing noise" added to the variance of the Gaussian process
    #
    # ~~~ For visualization (only applicable on 1d data)
    "MAKE_GIF" : True,
    "TITLE" : "title of my gif",            # ~~~ if MAKE_GIF is True, this will be the file name of the created .gif
    "HOW_OFTEN" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ for how many frames should the state after training be rendered
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ how many posterior predictive samples to plot
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "N_POSTERIOR_SAMPLES" : 100,            # ~~~ for plotting, posterior distributions are approximated as empirical dist.'s of this many samples
    #
    # ~~~ For metrics and visualization
    "EXTRA_STD" : True,
    "N_POSTERIOR_SAMPLES_EVALUATION" : 1000,# ~~~ for computing our model evaluation metrics, posterior distributions are approximated as empirical dist.'s of this many samples
    "SHOW_DIAGNOSTICS" : True,
    "SHOW_PLOT" : True
}

#
# ~~~ Define the variable `input_json_filename` (str, no default), along with `model_save_dir` (str, default None), and `final_test` and `overwrite_json` (both Bool, default False)
if hasattr(sys,"ps1"):
    #
    # ~~~ If this is an interactive (not srcipted) session, i.e., we are directly typing/pasting in the commands (I do this for debugging), then use the demo json name
    input_json_filename = "demo_bnn.json"
    model_save_dir = None
    final_test = False
    overwrite_json = False
else:
    #
    # ~~~ Use argparse to extract the file name `my_hyperparmeters.json` from `python train_bnn.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument( '--json', type=str, required=True )
    except:
        print("")
        print("    Hint: try `python train_bnn.py --json demo_bnn`")
        print("")
        raise
    parser.add_argument( '--model_save_dir', type=str )
    parser.add_argument( '--final_test', action=argparse.BooleanOptionalAction )
    parser.add_argument( '--overwrite_json', action=argparse.BooleanOptionalAction )
    args = parser.parse_args()
    model_save_dir = args.model_save_dir
    final_test = (args.final_test is not None)
    overwrite_json = (args.overwrite_json is not None)
    input_json_filename = args.json
    input_json_filename = input_json_filename if input_json_filename.endswith(".json") else input_json_filename+".json"

#
# ~~~ Load the .json file into a dictionary
hyperparameters = json_to_dict(input_json_filename)

#
# ~~~ Load the dictionary's key/value pairs into the global namespace
globals().update(hyperparameters)       # ~~~ e.g., if hyperparameters=={ "a":1, "B":2 }, then this defines a=1 and B=2

#
# ~~~ Might as well fix a seed, e.g., for randomly shuffling the order of batches during training
torch.manual_seed(SEED)

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
DTYPE = getattr(torch,DTYPE)            # ~~~ e.g., "float" (str) -> torch.float (torch.dtype) 
torch.set_default_dtype(DTYPE)
Optimizer = getattr(optim,OPTIMIZER)    # ~~~ e.g., OPTIMIZER=="Adam" (str) -> Optimizer==optim.Adam

#
# ~~~ Load the data
try:
    data = import_module(f"bnns.data.{DATA}")   # ~~~ this is equivalent to `import bnns.data.<DATA> as data`
except:
    data = import_module(DATA)                  # ~~~ this is equivalent to `import <DATA> as data` (works if DATA.py is in the cwd or anywhere on the path)

D_train = set_Dataset_attributes( data.D_train, device=DEVICE, dtype=DTYPE )
D_test  =  set_Dataset_attributes( data.D_test, device=DEVICE, dtype=DTYPE )
D_val   =   set_Dataset_attributes( data.D_val, device=DEVICE, dtype=DTYPE ) # ~~~ for hyperparameter evaulation and such, use the validation set instead of the "true" test set
data_is_univariate = (D_train[0][0].numel()==1)
x_train, y_train   =   convert_Dataset_to_Tensors(D_train)
x_test,  y_test    =   convert_Dataset_to_Tensors(D_test if final_test else D_val)

try:
    grid = data.grid.to( device=DEVICE, dtype=DTYPE )
except AttributeError:
    pass
except:
    raise

#
# ~~~ Load the network architecture
try:
    model = import_module(f"bnns.models.{MODEL}")   # ~~~ this is equivalent to `import bnns.models.<MODEL> as model`
except:
    model = import_module(MODEL)                    # ~~~ this is equivalent to `import <MODEL> as model` (works if MODEL.py is in the cwd or anywhere on the path)

BNN = model.BNN.to( device=DEVICE, dtype=DTYPE )
BNN.conditional_std = torch.tensor(CONDITIONAL_STD) # ~~~ relevant for all training methods
BNN.prior_J = PRIOR_J                               # ~~~ SSGE accuracy hyperparameter (only relevant for Sun et al. 2019)
BNN.post_J = POST_J                                 # ~~~ SSGE accuracyhyperparameter (only relevant for Sun et al. 2019)
BNN.prior_eta = PRIOR_eta                           # ~~~ stabilizing noise for SSGE (only relevant for Sun et al. 2019)
BNN.post_eta = POST_eta                             # ~~~ stabilizing noise for SSGE (only relevant for Sun et al. 2019)
BNN.prior_M = PRIOR_M                               # ~~~ SSGE accuracy hyperparameter (only relevant for Sun et al. 2019)
BNN.post_M = POST_M                                 # ~~~ SSGE accuracy hyperparameter (only relevant for Sun et al. 2019)
BNN.post_GP_eta = POST_GP_eta                       # ~~~ stabilizing noise for the GP approximation of the neural net (only relevant for Rudner et al. 2023, i.e., GAUSSIAN_APPROXIMATION==True)
try:
    assert DEFAULT_INITIALIZATION in ("new","old")
    BNN.set_default_uncertainty(DEFAULT_INITIALIZATION=="new")
except:
    BNN.projection_step( soft = not PROJECT )

if GP_PRIOR:
    from bnns.GPR import simple_mean_zero_RPF_kernel_GP as GP
    BNN.GP = GP( out_features=BNN.out_features, eta=GP_PRIOR_ETA )
    if not FUNCTIONAL:
        my_warn("The Gaussian process prior specified by `GP_PRIOR=True` will be ignored because `FUNCTIONAL==False`.")



### ~~~
## ~~~ Train a Bayesian neural network, either using BBB, or some functional mehtod
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( D_train, batch_size=BATCH_SIZE )
testloader = torch.utils.data.DataLoader( (D_test if final_test else D_val), batch_size=BATCH_SIZE )
optimizer = Optimizer( BNN.parameters(), lr=LR )
n_batches = len(dataloader)
n_test_batches = len(testloader)
n_params = sum( p.numel() for p in BNN.model_mean.parameters() )

#
# ~~~ Some naming stuff
description_of_the_experiment = "fBNN" if FUNCTIONAL else "BBB"
if GAUSSIAN_APPROXIMATION:
    if FUNCTIONAL:
        description_of_the_experiment += " Using a Gaussian Approximation"
    else:
        my_warn("The settings GAUSSIAN_APPROXIMATION=True and FUNCTIONAL=False are incompatible, since Rudner et al.'s Gaussian approximation is only used in fBNNs. The former will be ignored.")

#
# ~~~ Use the description_of_the_experiment as the title if no TITLE is specified
try:
    title = description_of_the_experiment if (TITLE is None) else TITLE
except NameError:
    title = description_of_the_experiment

#
# ~~~ Some plotting stuff
if data_is_univariate:
    #
    # ~~~ Define some objects used for plotting
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    #
    # ~~~ Define the main plotting routine
    plot_predictions = plot_bnn_empirical_quantiles if VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
    def plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, bnn, extra_std=(CONDITIONAL_STD if EXTRA_STD else 0.), how_many_individual_predictions=HOW_MANY_INDIVIDUAL_PREDICTIONS, n_posterior_samples=N_POSTERIOR_SAMPLES, title=title, prior=False ):
        #
        # ~~~ Draw from the posterior predictive distribuion
        with torch.no_grad():
            forward = bnn.prior_forward if prior else bnn
            predictions = torch.stack([ forward(grid,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES) ]).squeeze()
        return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, HOW_MANY_INDIVIDUAL_PREDICTIONS, title )
    #
    # ~~~ Plot the state of the posterior predictive distribution upon its initialization
    if MAKE_GIF:
        #
        # ~~~ Make the gif, and save `INITIAL_FRAME_REPETITIONS` copies of an identical image of the initial distribution
        gif = GifMaker(title)   # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN, prior=True )
        for j in range(INITIAL_FRAME_REPETITIONS):
            gif.capture( clear_frame_upon_capture=(j+1==INITIAL_FRAME_REPETITIONS) )

#
# ~~~ Establish some variables used for training
N_EPOCHS = non_negative_list( N_EPOCHS, integer_only=True ) # ~~~ supports N_EPOCHS to be a list of integers
STRIDE   = non_negative_list(  STRIDE,  integer_only=True ) # ~~~ supports STRIDE to be a list of integers
assert np.diff(N_EPOCHS+[N_EPOCHS[-1]+1]).min()>0, "The given sequence N_EPOCHS is not strictly increasing."
train_loss_curve = []
val_loss_curve = []
train_lik_curve = []
val_lik_curve = []
kl_div_curve = []
train_acc_curve = []
val_acc_curve = []
iter_count = []
epochs_completed_so_far = 0
target_epochs = N_EPOCHS.pop(0)
starting_time = time()
first_round = True
keep_training = True
min_val_loss = float("inf")
if EARLY_STOPPING:
    #
    # ~~~ Define all len(PATIENCE)*len(DELTA)*len(STRIDE) stopping conditions
    PATIENCE = non_negative_list( PATIENCE, integer_only=True )         # ~~~ supports PATIENCE to be a list of integers
    DELTA    = convert_to_list_and_check_items( DELTA, classes=float )  # ~~~ supports DELTA to be a list of integers
    stride_patience_and_delta_stopping_conditions = [
            [
                EarlyStopper( patience=patience, delta=delta )
                for delta, patience in product(DELTA,PATIENCE)
            ]
            for _ in STRIDE
        ]

#
# ~~~ Set "regularization parameters" for a Bayesian loss function (i.e., relative weights of the likelihood and the KL divergence)
if not isinstance(WEIGHTING,str):
    my_warn(f"Expected WEIGHTING to be a string, but found instead type(WEIGHTING)=={type(WEIGHTING)}. The loss function will be weighted as if WEIGHTING='standard'.")
elif WEIGHTING=="Blundell":
    #
    # ~~~ Follow the suggestion "\pi_i = \frac{2^{M-i}}{2^M-1}" from page 5 of https://arxiv.org/abs/1505.05424
    def decide_weights(**kwargs):
        i = kwargs["b"]
        M = kwargs["n_batches"]
        pi_i = 2**(M-i)/(2**M-1)
        return  pi_i, 1.
elif WEIGHTING=="Sun in principle": # (EQUIVALENT TO THE "standard" WEIGHTING BELOW)
    #
    # ~~~ Follow the suggestion "In principle, \lambda should be set as 1/|\mathcal{D}|" in equation (12) of https://arxiv.org/abs/1903.05779
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        D   = kwargs["D_train"]
        weight_on_the_kl         = 1/len(D)
        weight_on_the_likelihood = 1/len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood
elif WEIGHTING=="Sun in practice":
    #
    # ~~~ Follow the suggestion "We used \lambda=1/|\mathcal{D}_s| in practice" in equation (12) of https://arxiv.org/abs/1903.05779
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        weight_on_the_kl         = 1/len(D_s)
        weight_on_the_likelihood = 1/len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood
elif WEIGHTING=="naive":
    #
    # ~~~ Naively average the marginal KL divergences of each parameter, as well as the marginal likelihoods for each data point
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        n_params = kwargs["n_params"]
        weight_on_the_kl         = 1/n_params
        weight_on_the_likelihood = 1/len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood
else:
    #
    # ~~~ Downweight the KL divergence in the simplest manner possible to match the expectation of the minibatch estimator of likelihood
    decide_weights = lambda **kwargs: (1/n_batches, 1.)  # ~~~ this normalization achchieves an unbiased estimate of the variational loss
    if not WEIGHTING=="standard":
        my_warn(f'The given value of WEIGHTING ({WEIGHTING}) was not recognized. Using the default setting of WEIGHTING="standard" instead.')

#
# ~~~ One or two safety checks
if EXACT_WEIGHT_KL and FUNCTIONAL:
    my_warn("The settings EXACT_WEIGHT==True and FUNCTIONAL==True are incompatible. The former will be ignored.")

#
# ~~~ Do the actual training loop
while keep_training:
    with support_for_progress_bars():   # ~~~ this just supports green progress bars
        stopped_early = False
        pbar = tqdm( desc=description_of_the_experiment, total=target_epochs*len(dataloader), initial=epochs_completed_so_far*len(dataloader), ascii=' >=' )
        # ~~~ 
        #
        ### ~~~
        ## ~~~ Main Loop
        ### ~~~
        #
        # ~~~ The actual training logic (see train_nn.py for a simpler analogy)
        for e in range( target_epochs - epochs_completed_so_far ):
            for b, (X,y) in enumerate(dataloader):
                X, y = X.to(DEVICE), y.to(DEVICE)
                #
                # ~~~ Compute the gradient of the loss function on the batch (X,y)
                for j in range(N_MC_SAMPLES):
                    #
                    # ~~~ Draw a new sample
                    BNN.sample_from_standard_normal()
                    #
                    # ~~~ Compute the KL divergence of the (approximate) posterior against the user-specified prior
                    if not FUNCTIONAL:
                        kl_div = BNN.weight_kl(exact_formula=EXACT_WEIGHT_KL)
                    else:
                        BNN.sample_new_measurement_set()
                    if FUNCTIONAL and not GAUSSIAN_APPROXIMATION:
                        kl_div = BNN.functional_kl()
                    if FUNCTIONAL and GAUSSIAN_APPROXIMATION:
                        kl_div = BNN.gaussian_kl(approximate_mean=APPPROXIMATE_GAUSSIAN_MEAN)
                    #
                    # ~~~ Compute the likelihood term (this is the same for all training methods)
                    log_likelihood_density = BNN.log_likelihood_density(X,y)
                    #
                    # ~~~ Compute the loss==negative_ELBO
                    alpha, beta = decide_weights( b=b, n_batches=n_batches, X=X, D_train=D_train )
                    negative_ELBO = ( alpha*kl_div - beta*log_likelihood_density )/N_MC_SAMPLES
                    negative_ELBO.backward()
                #
                # ~~~ Perform the gradient-based update
                if not PROJECT:
                    BNN.apply_chain_rule_for_soft_projection()
                optimizer.step()
                optimizer.zero_grad()
                BNN.projection_step( soft = not PROJECT )
                #
                # ~~~ Report a moving average of train_loss as well as val_loss in the progress bar
                if len(train_loss_curve)>0:
                    pbar_info = { "train_loss": f"{ avg(train_loss_curve[-min(STRIDE):]) :<4.4f}" }
                    if len(val_loss_curve)>0:
                        pbar_info["val_loss"] = f"{ avg(val_loss_curve[-min(STRIDE):]) :<4.4f}"
                    if len(val_acc_curve)>0:
                        pbar_info["val_acc"] = f"{ avg(val_acc_curve[-min(STRIDE):]) :<4.4f}"
                    pbar.set_postfix(pbar_info)
                _ = pbar.update()
                #
                # ~~~ Every so often, do some additional stuff, too...
                if (pbar.n+1)%HOW_OFTEN==0:
                    #
                    # ~~~ Plotting logic
                    if data_is_univariate and MAKE_GIF:
                        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
                        gif.capture()
                    #
                    # ~~~ Record a little diagnostic info
                    with torch.no_grad():
                        #
                        # ~~~ Misc.
                        iter_count.append(pbar.n)
                        #
                        # ~~~ Diagnostic info specific to the last seen batch of training data
                        train_loss_curve.append(negative_ELBO.item())
                        predictions_train = torch.stack([ BNN(X,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES) ])
                        train_acc_curve.append(rmse_of_mean( predictions_train, y ))
                        train_lik_curve.append(log_likelihood_density.item())
                        #
                        # ~~~ Diagnostic info on the kl divergence between posterior and prior
                        kl_div = kl_div.item()
                        kl_div_curve.append(kl_div)
                        #
                        # ~~~ Diagnostic info specific to a randomly chosen batch of validation data
                        this_one = np.random.randint(n_test_batches)
                        for b, (X,y) in enumerate(testloader):
                            X, y = X.to(DEVICE), y.to(DEVICE)
                            if b==this_one:
                                val_lik = BNN.log_likelihood_density(X,y).item()
                                val_lik_curve.append(val_lik)
                                alpha, beta = decide_weights( b=b, n_batches=n_test_batches, X=X, D_train=D_train )
                                val_loss = alpha*kl_div - beta*val_lik
                                break
                        val_loss_curve.append(val_loss)
                        predictions_val = torch.stack([ BNN(X,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES) ])
                        val_acc_curve.append(rmse_of_mean( predictions_val, y ))
                        #
                        # ~~~ Save only the "best" parameters thus far
                        if val_loss < min_val_loss:
                            best_pars_so_far = BNN.state_dict()
                            best_iter_so_far = pbar.n
                            min_val_loss = val_loss
                    #
                    # ~~~ Assess whether or not any new stopping condition is triggered (although, training won't stop until *every* stopping condition is triggered)
                    if EARLY_STOPPING:
                        for i, stride in enumerate(STRIDE):
                            patience_and_delta_stopping_conditions = stride_patience_and_delta_stopping_conditions[i]
                            moving_avg_of_val_loss = avg(val_loss_curve[-stride:])
                            for j, early_stopper in enumerate(patience_and_delta_stopping_conditions):
                                stopped_early = early_stopper(moving_avg_of_val_loss)
                                if stopped_early:
                                    patience, delta = early_stopper.patience, early_stopper.delta
                                    del patience_and_delta_stopping_conditions[j]
                                    if all( len(lst)==0 for lst in stride_patience_and_delta_stopping_conditions ):
                                        keep_training = False
                                    break   # ~~~ break out of the loop over early stoppers
                            if stopped_early:
                                break       # ~~~ break out of the loop over strides
                    if stopped_early:
                        break               # ~~~ break out of the loop over batches
            if stopped_early:
                break                       # ~~~ break out of the loop over epochs
        total_iterations = pbar.n
        pbar.close()
        epochs_completed_so_far += e
        #
        # ~~~ If we reached the target number of epochs, then update `target_epochs` and do not record any early stopping hyperparameters
        if not stopped_early:
            epochs_completed_so_far += 1
            patience, delta, stride = None, None, None
            try:
                target_epochs = N_EPOCHS.pop(0)
            except IndexError:
                keep_training = False
        # ~~~
        #
        ### ~~~
        ## ~~~ Metrics (evaluate the model at this checkpoint, and save the results)
        ### ~~~
        #
        # ~~~ Define the predictive process
        def predict(loader,n):
            with torch.no_grad():
                data_is_unlabeled = isinstance( next(iter(loader)), torch.Tensor )
                predictions = []
                for _ in range(n):
                    BNN.sample_from_standard_normal()
                    predictions.append(torch.row_stack([
                                BNN( batch if data_is_unlabeled else batch[0], resample_weights=False )
                                for batch in loader
                            ]))
                predictions = torch.stack(predictions)
                if EXTRA_STD:
                    predictions += CONDITIONAL_STD*torch.randn_like(predictions)
                return predictions
        #
        # ~~~ Compute the posterior predictive distribution on the testing dataset(s)
        predictions = predict( testloader, N_POSTERIOR_SAMPLES_EVALUATION )
        try:
            interpolary_grid = data.interpolary_grid.to( device=DEVICE, dtype=DTYPE )
            extrapolary_grid = data.extrapolary_grid.to( device=DEVICE, dtype=DTYPE )
            batched_interpolary_grid = torch.utils.data.DataLoader( interpolary_grid, batch_size=BATCH_SIZE )
            batched_extrapolary_grid = torch.utils.data.DataLoader( extrapolary_grid, batch_size=BATCH_SIZE )
            predictions_on_interpolary_grid = predict( batched_interpolary_grid, N_POSTERIOR_SAMPLES_EVALUATION )
            predictions_on_extrapolary_grid = predict( batched_extrapolary_grid, N_POSTERIOR_SAMPLES_EVALUATION )
        except AttributeError:
            my_warn(f"Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}. For the best assessment of the quality of the UQ, please define these variables in the data file (no labels necessary)")
        #
        # ~~~ Compute the desired metrics
        hyperparameters["total_iter"] = total_iterations/len(dataloader)
        hyperparameters["best_iter"] = best_iter_so_far
        hyperparameters["epochs_completed"] = epochs_completed_so_far
        hyperparameters["compute_time"] = time() - starting_time
        hyperparameters["patience"] = patience
        hyperparameters["delta"] = delta
        hyperparameters["stride"] = stride
        hyperparameters["val_loss_curve"] = val_loss_curve
        hyperparameters["train_loss_curve"] = train_loss_curve
        hyperparameters["val_acc_curve"] = val_acc_curve
        hyperparameters["train_acc_curve"] = train_acc_curve
        hyperparameters["val_lik_curve"] = val_lik_curve
        hyperparameters["train_lik_curve"] = train_lik_curve
        hyperparameters["kl_div_curve"] = kl_div_curve
        hyperparameters["train_acc"] = avg(train_loss_curve[-min(STRIDE):])
        hyperparameters["METRIC_rmse_of_median"]             =      rmse_of_median( predictions, y_test )
        hyperparameters["METRIC_rmse_of_mean"]               =        rmse_of_mean( predictions, y_test )
        hyperparameters["METRIC_mae_of_median"]              =       mae_of_median( predictions, y_test )
        hyperparameters["METRIC_mae_of_mean"]                =         mae_of_mean( predictions, y_test )
        hyperparameters["METRIC_max_norm_of_median"]         =  max_norm_of_median( predictions, y_test )
        hyperparameters["METRIC_max_norm_of_mean"]           =    max_norm_of_mean( predictions, y_test )
        hyperparameters["METRIC_median_energy_score"]        =       energy_scores( predictions, y_test ).median().item()
        hyperparameters["METRIC_coverage"]                   =   aggregate_covarge( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES )
        hyperparameters["METRIC_median_avg_inverval_score"]  =  avg_interval_score_of_response_features( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES ).median().item()
        for use_quantiles in (True,False):
            show = SHOW_DIAGNOSTICS and (use_quantiles==VISUALIZE_DISTRIBUTION_USING_QUANTILES)  # ~~~ i.e., diagnostics are requesed, the prediction type mathces the uncertainty type (mean and std. dev., or median and iqr)
            tag = "quantile" if use_quantiles else "pm2_std"
            hyperparameters[f"METRIC_uncertainty_vs_accuracy_slope_{tag}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{tag}"]  =  uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES, quantile_accuracy=use_quantiles, show=show, verbose=SHOW_DIAGNOSTICS )
            try:
                hyperparameters[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{tag}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{tag}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, use_quantiles, extrapolary_grid, x_train, show=show, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation", verbose=SHOW_DIAGNOSTICS )
                hyperparameters[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{tag}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{tag}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, use_quantiles, interpolary_grid, x_train, show=show, title="Uncertainty vs Proximity to Data Within the Region of Interpolation", verbose=SHOW_DIAGNOSTICS )
                hyperparameters[f"METRIC_extrapolation_uncertainty_spread_{tag}"]  =  uncertainty_spread( predictions_on_extrapolary_grid, use_quantiles )
                hyperparameters[f"METRIC_interpolation_uncertainty_spread_{tag}"]  =  uncertainty_spread( predictions_on_interpolary_grid, use_quantiles )
            except NameError:
                pass    # ~~~ the user was already warned "Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}."
            except:
                raise
        #
        # ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data (the actual heatmaps)
        try:
            S = data.s_truncated.to( device=DEVICE, dtype=DTYPE )
            V = data.V_truncated.to( device=DEVICE, dtype=DTYPE )
            Y = data.unprocessed_y_test.to( device=DEVICE, dtype=DTYPE )
            def predict(loader,n):
                with torch.no_grad():
                    data_is_unlabeled = isinstance( next(iter(loader)), torch.Tensor )
                    predictions = torch.stack([
                            torch.row_stack([
                                    BNN( batch if data_is_unlabeled else batch[0], resample_weights=True )
                                    for batch in loader
                                ])
                            for _ in range(n)
                        ])
                    if EXTRA_STD:
                        predictions += CONDITIONAL_STD*torch.randn_like(predictions)
                    return predictions.mean(dim=0,keepdim=True) * S @ V.T
            predictions = predict( x_test, N_POSTERIOR_SAMPLES_EVALUATION )
            predictions_on_interpolary_grid = predict( batched_interpolary_grid, N_POSTERIOR_SAMPLES_EVALUATION )
            predictions_on_extrapolary_grid = predict( batched_extrapolary_grid, N_POSTERIOR_SAMPLES_EVALUATION )
            #
            # ~~~ Compute the desired metrics
            hyperparameters["METRIC_unprocessed_rmse_of_mean"]                =        rmse_of_mean( predictions, Y )
            hyperparameters["METRIC_unprocessed_rmse_of_median"]              =      rmse_of_median( predictions, Y )
            hyperparameters["METRIC_unprocessed_mae_of_mean"]                 =         mae_of_mean( predictions, Y )
            hyperparameters["METRIC_unprocessed_mae_of_median"]               =       mae_of_median( predictions, Y )
            hyperparameters["METRIC_unprocessed_max_norm_of_mean"]            =    max_norm_of_mean( predictions, Y )
            hyperparameters["METRIC_unprocessed_max_norm_of_median"]          =  max_norm_of_median( predictions, Y )
            hyperparameters["METRIC_unproccessed_coverage"]                   =   aggregate_covarge( predictions, Y, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES )
            hyperparameters["METRIC_unproccessed_median_energy_score"]        =       energy_scores( predictions, Y ).median().item()
            hyperparameters["METRIC_unproccessed_median_avg_inverval_score"]  =       avg_interval_score_of_response_features( predictions, Y, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES ).median().item()
            for estimator in ("mean","median"):
                hyperparameters[f"METRIC_unprocessed_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=SHOW_DIAGNOSTICS, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
                hyperparameters[f"METRIC_unprocessed_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=SHOW_DIAGNOSTICS, title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
                hyperparameters[f"METRIC_unprocessed_uncertainty_vs_accuracy_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"]                  =   uncertainty_vs_accuracy( predictions, Y, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES, quantile_accuracy=(estimator=="median"), show=SHOW_DIAGNOSTICS )
        except AttributeError:
            pass
        #
        # ~~~ Save the results
        if input_json_filename.startswith("demo"):
            my_warn(f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)')
        else:
            #
            # ~~~ Put together the output json filename
            output_json_filename = input_json_filename if overwrite_json else generate_json_filename()
            if first_round:
                first_round = False
                if overwrite_json:
                    os.remove(input_json_filename)
            output_json_filename = process_for_saving(output_json_filename)
            hyperparameters["filename"] = output_json_filename
            #
            # ~~~ Ok, now actually save the results
            if model_save_dir is not None:
                state_dict_path = os.path.join(
                        model_save_dir,
                        os.path.split(output_json_filename.strip(".json"))[1] + ".pth"
                    )
                hyperparameters["STATE_DICT_PATH"] = state_dict_path
                torch.save(
                        best_pars_so_far,
                        state_dict_path
                    )
            dict_to_json(
                    hyperparameters,
                    output_json_filename,
                    verbose = SHOW_DIAGNOSTICS
                )
        #
        # ~~~ Display the results
        if SHOW_DIAGNOSTICS:
            print_dict(hyperparameters)
        if SHOW_PLOT and keep_training and (not MAKE_GIF):
            fig,ax = plt.subplots(figsize=(12,6))
            fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
            plt.show()

#
# ~~~ Afterwards, develop the .gif or plot the trained model, if applicable
if data_is_univariate:
    if MAKE_GIF:
        for j in range(FINAL_FRAME_REPETITIONS):
            gif.frames.append( gif.frames[-1] )
        gif.develop(fps=24)
        plt.close()
    elif SHOW_PLOT:
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
        plt.show()

#
# ~~~ Validate implementation of the algorithm on the synthetic dataset "bivar_trivial"
if data.__name__ == "bnns.data.bivar_trivial":
    x_test = data.D_test.X.to( device=DEVICE, dtype=DTYPE )
    y_test = data.D_test.y.to( device=DEVICE, dtype=DTYPE )
    with torch.no_grad():
        predictions = torch.column_stack([ BNN(x_test,resample_weights=True).mean(dim=-1) for _ in range(N_POSTERIOR_SAMPLES_EVALUATION) ])
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot( x_test.cpu(), y_test.cpu(), "--", color="green" )
    y_pred = predictions.mean(dim=-1)
    plt.plot( x_test.cpu(), y_pred.cpu(), "-", color="blue" )
    fig.suptitle("If these lines roughly match, then the algorithm is surely working correctly")
    ax.grid()
    fig.tight_layout()
    plt.show()

#
# ~~~ A convenience function for diagnoistics when run interactively
def get_variable_name(obj,scope):
    return [name for name, value in scope.items() if value is obj]

def plot( lst, w=30, title=None ):
    assert len(lst)==len(iter_count)
    variable_name = get_variable_name( lst, globals() )[0]
    plt.plot( moving_average(iter_count,w), moving_average(lst,w) )
    plt.xlabel("Number of Iterations of Gradient Descent")
    plt.ylabel(variable_name)
    plt.suptitle( f"{variable_name} vs. #iter" if title is None else title )
    plt.grid()
    plt.tight_layout()
    plt.show()

if SHOW_DIAGNOSTICS:
    plot( kl_div_curve, title="KL Divergence as Training Progresses" )
    plot( train_loss_curve, title="Training Loss as Training Progresses" )
    plot( val_loss_curve, title="Validation Loss as Training Progresses" )
