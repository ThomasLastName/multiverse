
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
# ~~~ Package-specific utils
from bnns.utils import plot_nn, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, generate_json_filename, set_Dataset_attributes
from bnns.metrics import *

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
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
    "model" : "univar_NN",
    #
    # ~~~ For training
    "Optimizer" : "Adam",
    "lr" : 0.0005,
    "batch_size" : 64,
    "n_epochs" : 200,
    "n_MC_samples" : 1,                     # ~~~ relevant for droupout
    #
    # ~~~ For visualization
    "make_gif" : True,
    "how_often" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "initial_frame_repetitions" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "final_frame_repetitions" : 48,         # ~~~ for how many frames should the state after training be rendered
    "how_many_individual_predictions" : 6,  # ~~~ how many posterior predictive samples to plot
    "visualize_bnn_using_quantiles" : True, # ~~~ for dropout, if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "n_posterior_samples" : 100,            # ~~~ for dropout, how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "n_posterior_samples_evaluation" : 1000,
    "show_diagnostics" : True
}

#
# ~~~ Define the variable `input_json_filename`
if hasattr(sys,"ps1"):
    #
    # ~~~ If this is an interactive (not srcipted) session, i.e., we are directly typing/pasting in the commands (I do this for debugging), then use the demo json name
    input_json_filename = "demo_nn.json"
else:
    #
    # ~~~ Use argparse to extract the file name "my_hyperparmeters.json" from `python train_nn.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument( '--json', type=str, required=True )
    except:
        print("")
        print("    Hint: try `python train_nn.py --json demo_nn`")
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


#
# ~~~ Infer whether or not the model's forward pass is stochastic (e.g., whether or not it's using dropout)
X,_ = next(iter(torch.utils.data.DataLoader( D_train, batch_size=10 )))
with torch.no_grad():
    difference = NN(X)-NN(X)
    dropout = (difference.abs().mean()>0).item()



### ~~~
## ~~~ Train a conventional neural network, for reference
### ~~~

#
# ~~~ The optimizer, dataloader, and loss function
optimizer = Optimizer( NN.parameters(), lr=lr )
dataloader = torch.utils.data.DataLoader( D_train, batch_size=batch_size )
loss_fn = nn.MSELoss()

#
# ~~~ Some naming stuff
description_of_the_experiment = "Conventional, Deterministic Training" if not dropout else "Conventional Training of a Neural Network with Dropout"

#
# ~~~ Some plotting stuff
if data_is_univariate:
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    if dropout:
        #
        # ~~~ Override the plotting routine `plot_nn` by defining instead a routine which 
        plot_predictions = plot_bnn_empirical_quantiles if visualize_bnn_using_quantiles else plot_bnn_mean_and_std
        def plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, nn, extra_std=0., how_many_individual_predictions=how_many_individual_predictions, n_posterior_samples=n_posterior_samples, title=description_of_the_experiment ):
            #
            # ~~~ Draw from the predictive distribuion
            with torch.no_grad():
                predictions = torch.column_stack([ nn(grid) for _ in range(n_posterior_samples) ])
            return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, how_many_individual_predictions, title )
    #
    # ~~~ Plot the state of the model upon its initialization
    if make_gif:
        gif = GifMaker()      # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        for j in range(initial_frame_repetitions):
            gif.capture( clear_frame_upon_capture=(j+1==initial_frame_repetitions) )

with support_for_progress_bars():   # ~~~ this just supports green progress bars
    pbar = tqdm( desc=description_of_the_experiment, total=n_epochs*len(dataloader), ascii=' >=' )
    for e in range(n_epochs):
        #
        # ~~~ The actual training logic (totally conventional, hopefully familiar)
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if dropout:
                loss = 0.
                for _ in range(n_MC_samples):
                    loss += loss_fn(NN(X),y)/n_MC_samples
            else:
                loss = loss_fn(NN(X),y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({ "loss": f"{loss.item():<4.4f}" })
            _ = pbar.update()
        #
        # ~~~ Plotting logic
        if data_is_univariate and make_gif and (e+1)%how_often==0:
            fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
            gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)

pbar.close()

#
# ~~~ Afterwards, develop the .gif if applicable
if data_is_univariate:
    if make_gif:
        gif.develop( destination="NN", fps=24 )
        plt.close()
    else:
        fig,ax = plt.subplots(figsize=(12,6))
        fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        plt.show()

#
# ~~~ Validate implementation of the algorithm on the synthetic dataset "bivar_trivial"
if data.__name__ == "bnns.data.bivar_trivial":
    from bnns.data.univar_missing_middle import x_test, y_test
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot( x_test.cpu(), y_test.cpu(), "--", color="green" )
    with torch.no_grad():
        y_pred = NN(data.D_test.X.to( device=DEVICE, dtype=dtype )).mean(dim=-1)
    plt.plot( x_test.cpu(), y_pred.cpu(), "-", color="blue" )
    fig.suptitle("If these lines roughly match, then the algorithm is surely working correctly")
    ax.grid()
    fig.tight_layout()
    plt.show()



### ~~~
## ~~~ Metrics (evaluate the trained model)
### ~~~

#
# ~~~ Compute the posterior predictive distribution on the testing dataset
x_train, y_train  =  convert_Dataset_to_Tensors(D_train)
x_test, y_test    =    convert_Dataset_to_Tensors( D_test if final_test else D_val )
interpolary_grid = data.interpolary_grid.to( device=DEVICE, dtype=dtype )
extrapolary_grid = data.extrapolary_grid.to( device=DEVICE, dtype=dtype )

predict = lambda points: torch.stack([ NN(points) for _ in range(n_posterior_samples_evaluation) ]) if dropout else NN(points)

with torch.no_grad():
    predictions = predict(x_test)
    predictions_on_interpolary_grid = predict(interpolary_grid)
    predictions_on_extrapolary_grid = predict(extrapolary_grid)

#
# ~~~ Compute the desired metrics
if dropout:
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
else:
    hyperparameters["METRIC_mse"] = mse( NN, x_test, y_test )
    hyperparameters["METRIC_mae"] = mae( NN, x_test, y_test )
    hyperparameters["METRIC_max_norm"] = max_norm( NN, x_test, y_test )

#
# ~~~ Display the results
print_dict(hyperparameters)




### ~~~
## ~~~ Save the results
### ~~~

if input_json_filename.startswith("demo"):
    my_warn(f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)')
else:
    output_json_filename = input_json_filename if overwrite_json else generate_json_filename()
    dict_to_json( hyperparameters, output_json_filename )
    if model_save_dir is not None:
        # save the model, assuming model_save_dir could be something like `subfolder_of_experiments/model_name.pt`
        raise NotImplementedError("TODO")


#