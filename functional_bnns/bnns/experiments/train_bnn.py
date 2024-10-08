
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
    "dtype" : "float",
    "seed" : 2024,
    #
    # ~~~ Which problem
    "data" : "univar_missing_middle",
    "model" : "univar_BNN",
    #
    # ~~~ For training
    "gaussian_approximation" : True,    # ~~~ in an fBNN use a first order Gaussian approximation like Rudner et al.
    "functional" : False,   # ~~~ whether or to do functional training or (if False) BBB
    "n_MC_samples" : 20,    # ~~~ expectations (in the variational loss) are estimated as an average of this many Monte-Carlo samples
    "project" : True,       # ~~~ if True, use projected gradient descent; else use the weird thing from the paper
    "projection_tol" : 1e-6,# ~~~ for numerical reasons, project onto [projection_tol,Inf), rather than onto [0,Inft)
    "prior_J"   : 100,      # ~~~ `J` in the SSGE of the prior score
    "post_J"    : 10,       # ~~~ `J` in the SSGE of the posterior score
    "prior_eta" : 0.5,      # ~~~ `eta` in the SSGE of the prior score
    "post_eta"  : 0.5,      # ~~~ `eta` in the SSGE of the posterior score
    "prior_M"   : 4000,     # ~~~ `M` in the SSGE of the prior score
    "post_M"    : 40,       # ~~~ `M` in the SSGE of the posterior score
    "conditional_std" : 0.19,
    "Optimizer" : "Adam",
    "lr" : 0.0005,
    "batch_size" : 64,
    "n_epochs" : 200,
    #
    # ~~~ For visualization (only applicable on 1d data)
    "make_gif" : True,
    "how_often" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "initial_frame_repetitions" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "final_frame_repetitions" : 48,         # ~~~ for how many frames should the state after training be rendered
    "how_many_individual_predictions" : 6,  # ~~~ how many posterior predictive samples to plot
    "visualize_bnn_using_quantiles" : True, # ~~~ if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "n_posterior_samples" : 100,            # ~~~ for plotting, posterior distributions are approximated as empirical dist.'s of this many samples
    #
    # ~~~ For metrics and visualization
    "extra_std" : True,
    "n_posterior_samples_evaluation" : 1000,# ~~~ for computing our model evaluation metrics, posterior distributions are approximated as empirical dist.'s of this many samples
	"show_diagnostics" : True
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
    parser.add_argument( '--final_test', action=argparse.BooleanOptionalAction )
    parser.add_argument( '--overwrite_json', action=argparse.BooleanOptionalAction )
    args = parser.parse_args()
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
torch.manual_seed(seed)

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
dtype = getattr(torch,dtype)            # ~~~ e.g., "float" (str) -> torch.float (torch.dtype) 
torch.set_default_dtype(dtype)
Optimizer = getattr(optim,Optimizer)    # ~~~ e.g., "Adam" (str) -> optim.Adam

#
# ~~~ Load the network architecture
try:
    model = import_module(f"bnns.models.{model}")   # ~~~ this is equivalent to `import bnns.models.<model> as model`
except:
    model = import_module(model)

BNN = model.BNN.to( device=DEVICE, dtype=dtype )
BNN.conditional_std = torch.tensor(conditional_std)
BNN.prior_J = prior_J
BNN.post_J = post_J
BNN.prior_eta = prior_eta
BNN.post_eta = post_eta
BNN.prior_M = prior_M
BNN.post_M = post_M

#
# ~~~ Load the data
try:
    data = import_module(f"bnns.data.{data}")   # ~~~ this is equivalent to `import bnns.data.<data> as data`
except:
    data = import_module(data)

D_train = set_Dataset_attributes( data.D_train, device=DEVICE, dtype=dtype )
D_test  =  set_Dataset_attributes( data.D_test, device=DEVICE, dtype=dtype )
D_val   =   set_Dataset_attributes( data.D_val, device=DEVICE, dtype=dtype ) # ~~~ for hyperparameter evaulation and such, use the validation set instead of the "true" test set
data_is_univariate = (D_train[0][0].numel()==1)

try:
    scale = data.scale
    conditional_std /= scale
except:
    pass

try:
    grid = data.grid.to( device=DEVICE, dtype=dtype )
except:
    pass



### ~~~
## ~~~ Do Bayesian training
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( D_train, batch_size=batch_size )
mean_optimizer = Optimizer( BNN.model_mean.parameters(), lr=lr )
std_optimizer  =  Optimizer( BNN.model_std.parameters(), lr=lr )

#
# ~~~ Some naming stuff
description_of_the_experiment = "fBNN" if functional else "BBB"
if gaussian_approximation:
    if functional:
        description_of_the_experiment += " Using a Gaussian Approximation"
    else:
        my_warn("`gaussian_approximation` was specified as True, but `functional` was specified as False; since Rudner et al.'s Gaussian approximation is only used in fBNNs, it will not be used in this case.")

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
    plot_predictions = plot_bnn_empirical_quantiles if visualize_bnn_using_quantiles else plot_bnn_mean_and_std
    def plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, bnn, extra_std=(BNN.conditional_std if extra_std else 0.), how_many_individual_predictions=how_many_individual_predictions, n_posterior_samples=n_posterior_samples, title=description_of_the_experiment, prior=False ):
        #
        # ~~~ Draw from the posterior predictive distribuion
        with torch.no_grad():
            forward = bnn.prior_forward if prior else bnn
            predictions = torch.column_stack([ forward(grid,resample_weights=True) for _ in range(n_posterior_samples) ])
        return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, how_many_individual_predictions, title )
    #
    # ~~~ Plot the state of the posterior predictive distribution upon its initialization
    if make_gif:
        gif = GifMaker()      # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN, prior=True )
        for j in range(initial_frame_repetitions):
            gif.capture( clear_frame_upon_capture=(j+1==initial_frame_repetitions) )

#
# ~~~ Define some objects for recording the hisorty of training
metrics = ( "ELBO", "post", "prior", "like" )
history = {}
for metric in metrics:
    history[metric] = []

#
# ~~~ Define how to project onto the constraint set
if project:
    BNN.rho = lambda x:x
    def projection_step(BNN):
        with torch.no_grad():
            for p in BNN.model_std.parameters():
                p.data = torch.clamp( p.data, min=projection_tol )
    projection_step(BNN)

#
# ~~~ Define the measurement set for functional training
x_train, _ = convert_Dataset_to_Tensors(D_train)
BNN.measurement_set = x_train

#
# ~~~ Start the training loop
with support_for_progress_bars():   # ~~~ this just supports green progress bars
    pbar = tqdm( desc=description_of_the_experiment, total=n_epochs*len(dataloader), ascii=' >=' )
    for e in range(n_epochs):
        #
        # ~~~ Training logic
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            for j in range(n_MC_samples):
                #
                # ~~~ Compute the gradient of the loss function
                if functional:
                    if gaussian_approximation:
                        log_posterior_density = BNN.gaussian_kl( resample_measurement_set=False, add_stabilizing_noise=True )
                        log_prior_density = torch.tensor(0.)
                    else:
                        log_posterior_density, log_prior_density = BNN.functional_kl(resample_measurement_set=False)
                else:
                    BNN.sample_from_standard_normal()   # ~~~ draw a new Monte-Carlo sample for estimating the integrals as an MC average
                    log_posterior_density   =   BNN.log_posterior_density()
                    log_prior_density       =   BNN.log_prior_density()
            #
            # ~~~ Add the the likelihood term and differentiate
            log_likelihood_density = BNN.log_likelihood_density(X,y)
            negative_ELBO = ( log_posterior_density - log_prior_density - log_likelihood_density )/n_MC_samples
            negative_ELBO.backward()
            #
            # ~~~ This would be training based only on the data:
            # loss = -BNN.log_likelihood_density(X,y)
            # loss.backward()
            #
            # ~~~ Do the gradient-based update
            for optimizer in (mean_optimizer,std_optimizer):
                optimizer.step()
                optimizer.zero_grad()
            #
            # ~~~ Do the projection
            if project:
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
            m = X.shape[0]
            accuracy = -( log_likelihood_density.item() + (m/2)*torch.log(2*torch.pi*BNN.conditional_std) )/2 * BNN.conditional_std/m # ~~~ basically, mse if weights are Gaussian, mae if weights are Laplace, etc. maybe off by a factor of 2 or something
            to_print = { "conventional loss" : f"{accuracy.item():<4.4f}" }
            pbar.set_postfix(to_print)
            _ = pbar.update()
        #
        # ~~~ Plotting logic
        if data_is_univariate and make_gif and (e+1)%how_often==0:
            fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
            gif.capture()
            # print("captured")

pbar.close()

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if data_is_univariate:
    if make_gif:
        for j in range(final_frame_repetitions):
            gif.frames.append( gif.frames[-1] )
        gif.develop( destination=description_of_the_experiment, fps=24 )
        plt.close()
    else:
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
        plt.show()

#
# ~~~ Validate implementation of the algorithm on the synthetic dataset "bivar_trivial"
if data.__name__ == "bnns.data.bivar_trivial":
    x_test = data.D_test.X.to( device=DEVICE, dtype=dtype )
    y_test = data.D_test.y.to( device=DEVICE, dtype=dtype )
    with torch.no_grad():
        predictions = torch.column_stack([ BNN(x_test,resample_weights=True).mean(dim=-1) for _ in range(n_posterior_samples_evaluation) ])
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

# def plot( metric, window_size=n_epochs/50 ):
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
interpolary_grid = data.interpolary_grid.to( device=DEVICE, dtype=dtype )
extrapolary_grid = data.extrapolary_grid.to( device=DEVICE, dtype=dtype )

def predict(x):
        predictions = torch.stack([ BNN(x,resample_weights=True) for _ in range(n_posterior_samples_evaluation) ])
        if extra_std:
            predictions += BNN.conditional_std*torch.randn_like(predictions)
        return predictions

with torch.no_grad():
    predictions = predict(x_test)
    predictions_on_interpolary_grid = predict(interpolary_grid)
    predictions_on_extrapolary_grid = predict(extrapolary_grid)

#
# ~~~ Compute the desired metrics
hyperparameters["METRIC_mse_of_mean"]    =    mse_of_mean( predictions, y_test )
hyperparameters["METRIC_mae_of_mean"]    =    mae_of_mean( predictions, y_test )
hyperparameters["METRIC_max_norm_of_mean"]    =    max_norm_of_mean( predictions, y_test )
for estimator in ("mean","median"):
    hyperparameters[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
    hyperparameters[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
    hyperparameters[f"METRIC_uncertainty_vs_accuracy_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"]    =    uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty=visualize_bnn_using_quantiles, quantile_accuracy=(estimator=="median"), show=show_diagnostics )

#
# ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data (the actual heatmaps)
try:
    S = data.s_truncated.to( device=DEVICE, dtype=dtype )
    V = data.V_truncated.to( device=DEVICE, dtype=dtype )
    Y = data.unprocessed_y_test.to( device=DEVICE, dtype=dtype )
    def predict(x):
        predictions = torch.stack([ BNN(x,resample_weights=True) for _ in range(n_posterior_samples_evaluation) ])
        if extra_std:
            predictions += BNN.conditional_std*torch.randn_like(predictions)
        return predictions.mean(dim=0,keepdim=True) * S @ V.T
    with torch.no_grad():
        predictions = predict(x_test)
        predictions_on_interpolary_grid = predict(interpolary_grid)
        predictions_on_extrapolary_grid = predict(extrapolary_grid)
    #
    # ~~~ Compute the desired metrics
    hyperparameters["METRIC_unprocessed_mse_of_mean"]    =    mse_of_mean( predictions, Y )
    hyperparameters["METRIC_unprocessed_mae_of_mean"]    =    mae_of_mean( predictions, Y )
    hyperparameters["METRIC_unprocessed_max_norm_of_mean"]    =    max_norm_of_mean( predictions, Y )
    for estimator in ("mean","median"):
        hyperparameters[f"METRIC_unprocessed_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
        hyperparameters[f"METRIC_unprocessed_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
        hyperparameters[f"METRIC_unprocessed_uncertainty_vs_accuracy_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"]    =    uncertainty_vs_accuracy( predictions, Y, quantile_uncertainty=visualize_bnn_using_quantiles, quantile_accuracy=(estimator=="median"), show=show_diagnostics )
except:
    pass

#
# ~~~ Print the results
print_dict(hyperparameters)



### ~~~
## ~~~ Save the results
### ~~~

if input_json_filename.startswith("demo"):
    my_warn(f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)')
else:
    output_json_filename = input_json_filename if overwrite_json else generate_json_filename()
    dict_to_json( hyperparameters, output_json_filename )

#
