
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
from bnns.Stein_GD import SequentialSteinEnsemble as Ensemble
from bnns.metrics import *

#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, set_Dataset_attributes, generate_json_filename

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_torch_utils         import nonredundant_copy_of_module_list, convert_Dataset_to_Tensors
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, print_dict, my_warn



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
    "model" : "univar_NN",
    #
    # ~~~ For training
    "conditional_std" : 0.19,
    "bw" : None,
    "n_Stein_particles" : 100,
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
    input_json_filename = "demo_stein.json"
else:
    #
    # ~~~ Use argparse to extract the file name name "my_hyperparmeters.json" from `python train_stein.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument( '--json', type=str, required=True )
    except:
        print("")
        print("    Hint: try `python train_stein.py --json demo_stein`")
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

NN = model.NN.to( device=DEVICE, dtype=dtype )

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
## ~~~ Do a Stein neural network ensemble
### ~~~

#
# ~~~ Instantiate an ensemble
ensemble = Ensemble(
        architecture = nonredundant_copy_of_module_list(NN),
        n_copies = n_Stein_particles,
        Optimizer = lambda params: Optimizer( params, lr=lr ),
        conditional_std = torch.tensor(conditional_std),
        device = DEVICE,
        bw = bw
    )

#
# ~~~ The dataloader
dataloader = torch.utils.data.DataLoader( D_train, batch_size=batch_size )

#
# ~~~ Some plotting stuff
description_of_the_experiment = "Stein Neural Network Ensemble"
if data_is_univariate:
    #
    # ~~~ Define some objects used for plotting
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    #
    # ~~~ Define the main plotting routine
    plot_predictions = plot_bnn_empirical_quantiles if visualize_bnn_using_quantiles else plot_bnn_mean_and_std
    def plot_ensemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble, extra_std=(conditional_std if extra_std else 0.), how_many_individual_predictions=how_many_individual_predictions, title=description_of_the_experiment ):
        #
        # ~~~ Draw from the posterior predictive distribuion
        with torch.no_grad():
            predictions = ensemble(grid).squeeze().T
        return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, how_many_individual_predictions, title )
    #
    # ~~~ Plot the state of the posterior predictive distribution upon its initialization
    if make_gif:
        gif = GifMaker()      # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_ensemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )
        for j in range(initial_frame_repetitions):
            gif.capture( clear_frame_upon_capture=(j+1==initial_frame_repetitions) )

#
# ~~~ Do the actual training loop
with support_for_progress_bars():   # ~~~ this just supports green progress bars
    pbar = tqdm( desc=description_of_the_experiment, total=n_epochs*len(dataloader), ascii=' >=' )
    for e in range(n_epochs):
        #
        # ~~~ Training logic
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            ensemble.train_step(X,y)
            _ = pbar.update()
        with torch.no_grad():
            predictions = ensemble(X)
        pbar.set_postfix({ "mse of mean": f"{mse_of_mean(predictions,y):<4.4f}" })
        #
        # ~~~ Plotting logic
        if data_is_univariate and make_gif and (e+1)%how_often==0:
            fig,ax = plot_ensemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )
            gif.capture()
            # print("captured")

pbar.close()

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if data_is_univariate:
    if not make_gif:    # ~~~ make a plot now
        fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = plot_ensemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )
    if make_gif:
        for j in range(final_frame_repetitions):
            gif.capture( clear_frame_upon_capture=(j+1==final_frame_repetitions) )
        gif.develop( destination=description_of_the_experiment, fps=24 )
        plt.close()
    else:
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
    predictions = ensemble(x)
    if extra_std:
        predictions += ensemble.conditional_std*torch.randn_like(predictions)
    return predictions

with torch.no_grad():
    predictions = predict(x_test)
    predictions_on_interpolary_grid = predict(interpolary_grid)
    predictions_on_extrapolary_grid = predict(extrapolary_grid)

#
# ~~~ Compute the desired metrics
hyperparameters["METRIC_mse_of_median"]  =  mse_of_median( predictions, y_test )
hyperparameters["METRIC_mse_of_mean"]    =    mse_of_mean( predictions, y_test )
hyperparameters["METRIC_mae_of_median"]  =  mae_of_median( predictions, y_test )
hyperparameters["METRIC_mae_of_mean"]    =    mae_of_mean( predictions, y_test )
hyperparameters["METRIC_max_norm_of_median"]  =  max_norm_of_median( predictions, y_test )
hyperparameters["METRIC_max_norm_of_mean"]    =    max_norm_of_mean( predictions, y_test )
for estimator in ("mean","median"):
    hyperparameters[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
    hyperparameters[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=show_diagnostics, title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
    hyperparameters[f"METRIC_uncertainty_vs_accuracy_slope_{estimator}"], hyperparameters[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"]    =    uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty=visualize_bnn_using_quantiles, quantile_accuracy=(estimator=="median"), show=show_diagnostics )

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
