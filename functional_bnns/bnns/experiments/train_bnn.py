
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from torch import nn, optim
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from importlib import import_module
from time import time
import argparse
import sys

#
# ~~~ The guts of the model
from bnns.SequentialGaussianBNN import SequentialGaussianBNN
#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, set_Dataset_attributes, generate_json_filename
from bnns.metrics import *

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, print_dict, my_warn
from quality_of_life.my_torch_utils         import convert_Dataset_to_Tensors



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
    "FUNCTIONAL" : False,   # ~~~ whether or to do functional training or (if False) BBB
    "N_MC_SAMPLES" : 20,    # ~~~ expectations (in the variational loss) are estimated as an average of this many Monte-Carlo samples
    "PROJECT" : True,       # ~~~ if True, use projected gradient descent; else use the weird thing from the paper
    "PROJECTION_TOL" : 1e-6,# ~~~ for numerical reasons, project onto [PROJECTION_TOL,Inf), rather than onto [0,Inft)
    "PRIOR_J"   : 100,      # ~~~ `J` in the SSGE of the prior score
    "POST_J"    : 10,       # ~~~ `J` in the SSGE of the posterior score
    "PRIOR_eta" : 0.5,      # ~~~ `eta` in the SSGE of the prior score
    "POST_eta"  : 0.5,      # ~~~ `eta` in the SSGE of the posterior score
    "PRIOR_M"   : 4000,     # ~~~ `M` in the SSGE of the prior score
    "POST_M"    : 40,       # ~~~ `M` in the SSGE of the posterior score
    "POST_GP_eta" : 0.001,  # ~~~ the level of the "stabilizing noise" added to the Gaussian approximation of the posterior distribution if `gaussian_approximation` is True
    "CONDITIONAL_STD" : 0.19,
    "OPTIMIZER" : "Adam",
    "LR" : 0.0005,
    "BATCH_SIZE" : 64,
    "N_EPOCHS" : 200,
    #
    # ~~~ For visualization (only applicable on 1d data)
    "MAKE_GIF" : True,
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
	"SHOW_DIAGNOSTICS" : True
}

#
# ~~~ Define the variable `input_json_filename`
if hasattr(sys,"ps1"):
    #
    # ~~~ If this is an interactive (not srcipted) session, i.e., we are directly typing/pasting in the commands (I do this for debugging), then use the demo json name
    input_json_filename = "demo_bnn.json"
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

try:
    scale = data.scale
    CONDITIONAL_STD /= scale
except:
    pass

try:
    grid = data.grid.to( device=DEVICE, dtype=DTYPE )
except:
    pass

#
# ~~~ Load the network architecture
try:
    model = import_module(f"bnns.models.{MODEL}")   # ~~~ this is equivalent to `import bnns.models.<MODEL> as model`
except:
    model = import_module(MODEL)                    # ~~~ this is equivalent to `import <MODEL> as model` (works if MODEL.py is in the cwd or anywhere on the path)

BNN = model.BNN.to( device=DEVICE, dtype=DTYPE )
BNN.conditional_std = torch.tensor(CONDITIONAL_STD)
BNN.prior_J = PRIOR_J
BNN.post_J = POST_J
BNN.prior_eta = PRIOR_eta
BNN.post_eta = POST_eta
BNN.prior_M = PRIOR_M
BNN.post_M = POST_M
BNN.post_GP_eta = POST_GP_eta



### ~~~
## ~~~ Do Bayesian training
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( D_train, batch_size=BATCH_SIZE )
optimizer = Optimizer( BNN.parameters(), lr=LR )

#
# ~~~ Some naming stuff
description_of_the_experiment = "fBNN" if FUNCTIONAL else "BBB"
if GAUSSIAN_APPROXIMATION:
    if FUNCTIONAL:
        description_of_the_experiment += " Using a Gaussian Approximation"
    else:
        my_warn("`GAUSSIAN_APPROXIMATION` was specified as True, but `FUNCTIONAL` was specified as False; since Rudner et al.'s Gaussian approximation is only used in fBNNs, it will not be used in this case.")

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
    def plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, bnn, extra_std=(CONDITIONAL_STD if EXTRA_STD else 0.), how_many_individual_predictions=HOW_MANY_INDIVIDUAL_PREDICTIONS, n_posterior_samples=N_POSTERIOR_SAMPLES, title=description_of_the_experiment, prior=False ):
        #
        # ~~~ Draw from the posterior predictive distribuion
        with torch.no_grad():
            forward = bnn.prior_forward if prior else bnn
            predictions = torch.stack([ forward(grid,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES) ]).squeeze()
        return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, HOW_MANY_INDIVIDUAL_PREDICTIONS, title )
    #
    # ~~~ Plot the state of the posterior predictive distribution upon its initialization
    if MAKE_GIF:
        gif = GifMaker()      # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN, prior=True )
        for j in range(INITIAL_FRAME_REPETITIONS):
            gif.capture( clear_frame_upon_capture=(j+1==INITIAL_FRAME_REPETITIONS) )

#
# ~~~ Define some objects for recording the hisorty of training
metrics = ( "ELBO", "post", "prior", "like" )
history = {}
for metric in metrics:
    history[metric] = []

#
# ~~~ Define how to project onto the constraint set
if PROJECT:
    BNN.rho = lambda x:x
    def projection_step(BNN):
        with torch.no_grad():
            for p in BNN.model_std.parameters():
                p.data = torch.clamp( p.data, min=PROJECTION_TOL )
    projection_step(BNN)

#
# ~~~ Define the measurement set for functional training
x_train, _ = convert_Dataset_to_Tensors(D_train)
BNN.measurement_set = x_train

#
# ~~~ Start the training loop
with support_for_progress_bars():   # ~~~ this just supports green progress bars
    pbar = tqdm( desc=description_of_the_experiment, total=N_EPOCHS*len(dataloader), ascii=' >=' )
    starting_time = time()
    for e in range(N_EPOCHS):
        #
        # ~~~ Training logic
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            for j in range(N_MC_SAMPLES):
                #
                # ~~~ Compute the gradient of the loss function
                BNN.sample_from_standard_normal()   # ~~~ draw a new MC sample for estimating the integrals
                if not FUNCTIONAL:
                    kl_div = BNN.weight_kl()
                if FUNCTIONAL and not GAUSSIAN_APPROXIMATION:
                    kl_div = BNN.functional_kl()
                if FUNCTIONAL and GAUSSIAN_APPROXIMATION:
                    kl_div = BNN.gaussian_kl(approximate_mean=APPPROXIMATE_GAUSSIAN_MEAN)
            #
            # ~~~ Add the the likelihood term and differentiate
            log_likelihood_density = BNN.log_likelihood_density(X,y)
            negative_ELBO = ( kl_div - log_likelihood_density )/N_MC_SAMPLES
            negative_ELBO.backward()
            #
            # ~~~ This would be training based only on the data:
            # loss = -BNN.log_likelihood_density(X,y)
            # loss.backward()
            #
            # ~~~ Do the gradient-based update
            optimizer.step()
            optimizer.zero_grad()
            #
            # ~~~ Do the projection
            if PROJECT:
                projection_step(BNN)
            #
            # ~~~ Record some diagnostics
            # history["ELBO"].append( -negative_ELBO.item())
            # history["post"].append( log_posterior_density.item())
            # history["prior"].append(log_prior_density.item())
            # history["like"].append( log_likelihood_density.item())
            # to_print = {
            #     "ELBO" : f"{-negative_ELBO.item():<4.4f}",
            #     "post" : f"{log_posterior_density.item():<4.4f}",
            #     "prior": f"{log_prior_density.item():<4.4f}",
            #     "like" : f"{log_likelihood_density.item():<4.4f}"
            # }
            _ = pbar.update()
        with torch.no_grad():
            predictions = torch.stack([ BNN(X,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES) ])
            to_print = { "rmse_of_mean" : f"{rmse_of_mean(predictions,y):<4.4f}" }
        pbar.set_postfix(to_print)
        #
        # ~~~ Plotting logic
        if data_is_univariate and MAKE_GIF and (e+1)%HOW_OFTEN==0:
            fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
            gif.capture()
            # print("captured")

pbar.close()
ending_time = time()

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if data_is_univariate:
    if MAKE_GIF:
        for j in range(FINAL_FRAME_REPETITIONS):
            gif.frames.append( gif.frames[-1] )
        gif.develop( destination=description_of_the_experiment, fps=24 )
        plt.close()
    else:
        if SHOW_DIAGNOSTICS:
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



### ~~~
## ~~~ Debugging diagnostics
### ~~~

# def plot( metric, window_size=N_EPOCHS/50 ):
#     plt.plot( moving_average(history[metric],int(window_size)) )
#     plt.grid()
#     plt.tight_layout()
#     plt.show()



### ~~~
## ~~~ Metrics (evaluate the trained model)
### ~~~

#
# ~~~ Compute the posterior predictive distribution on the testing dataset
x_train, y_train  =  convert_Dataset_to_Tensors(D_train)
x_test, y_test    =    convert_Dataset_to_Tensors( D_test if final_test else D_val )

def predict(x):
        predictions = torch.stack([ BNN(x,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES_EVALUATION) ])
        if EXTRA_STD:
            predictions += CONDITIONAL_STD*torch.randn_like(predictions)
        return predictions

with torch.no_grad():
    predictions = predict(x_test)

try:
    interpolary_grid = data.interpolary_grid.to( device=DEVICE, dtype=DTYPE )
    extrapolary_grid = data.extrapolary_grid.to( device=DEVICE, dtype=DTYPE )        
    predictions_on_interpolary_grid = predict(interpolary_grid)
    predictions_on_extrapolary_grid = predict(extrapolary_grid)
except AttributeError:
    my_warn(f"Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}. For the best assessment of the quality of the UQ, please define these variables in the data file (no labels necessary)")

#
# ~~~ Compute the desired metrics
hyperparameters["METRIC_compute_time"] = ending_time - starting_time
hyperparameters["METRIC_rmse_of_median"]             =      rmse_of_median( predictions, y_test )
hyperparameters["METRIC_rmse_of_mean"]               =        rmse_of_mean( predictions, y_test )
hyperparameters["METRIC_mae_of_median"]              =       mae_of_median( predictions, y_test )
hyperparameters["METRIC_mae_of_mean"]                =         mae_of_mean( predictions, y_test )
hyperparameters["METRIC_max_norm_of_median"]         =  max_norm_of_median( predictions, y_test )
hyperparameters["METRIC_max_norm_of_mean"]           =    max_norm_of_mean( predictions, y_test )
hyperparameters["METRIC_median_energy_score"]        =       energy_scores( predictions, y_test ).median().item()
hyperparameters["METRIC_coverage"]                   =   aggregate_covarge( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES )
hyperparameters["METRIC_median_avg_inverval_score"]  =  avg_interval_score_of_response_features( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES ).median().item()
for estimator in ("mean","median"):
    hyperparameters[f"METRIC_uncertainty_vs_accuracy_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"] = uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty=VISUALIZE_DISTRIBUTION_USING_QUANTILES, quantile_accuracy=(estimator=="median"), show=SHOW_DIAGNOSTICS )
    try:
        hyperparameters[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=SHOW_DIAGNOSTICS, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
        hyperparameters[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=SHOW_DIAGNOSTICS, title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
    except NameError:
        pass

#
# ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data (the actual heatmaps)
try:
    S = data.s_truncated.to( device=DEVICE, dtype=DTYPE )
    V = data.V_truncated.to( device=DEVICE, dtype=DTYPE )
    Y = data.unprocessed_y_test.to( device=DEVICE, dtype=DTYPE )
    def predict(x):
        predictions = torch.stack([ BNN(x,resample_weights=True) for _ in range(N_POSTERIOR_SAMPLES_EVALUATION) ])
        if EXTRA_STD:
            predictions += CONDITIONAL_STD*torch.randn_like(predictions)
        return predictions.mean(dim=0,keepdim=True) * S @ V.T
    with torch.no_grad():
        predictions = predict(x_test)
        predictions_on_interpolary_grid = predict(interpolary_grid)
        predictions_on_extrapolary_grid = predict(extrapolary_grid)
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
# ~~~ Print the results
if SHOW_DIAGNOSTICS:
    print_dict(hyperparameters)



### ~~~
## ~~~ Save the results
### ~~~

if input_json_filename.startswith("demo"):
    my_warn(f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)')
else:
    output_json_filename = input_json_filename if overwrite_json else generate_json_filename()
    if model_save_dir is not None:
        hyperparameters["MODEL_SAVE_DIR"] = model_save_dir
        raise NotImplementedError("TODO")
    dict_to_json( hyperparameters, output_json_filename, override=overwrite_json, verbose=SHOW_DIAGNOSTICS )


#
