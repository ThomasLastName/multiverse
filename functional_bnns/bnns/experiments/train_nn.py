
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import numpy as np
import torch
from torch import nn, optim
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
from bnns.utils import plot_nn, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, generate_json_filename, set_Dataset_attributes, non_negative_list, EarlyStopper
from bnns.metrics import *

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, print_dict, my_warn, process_for_saving
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
    "MODEL" : "univar_NN",
    #
    # ~~~ For training
    "OPTIMIZER" : "Adam",
    "LR" : 0.0005,
    "BATCH_SIZE" : 64,
    "N_EPOCHS" : 200,
    "EARLY_STOPPING" : True,
    "DELTA": 0.05,
    "PATIENCE" : 20,
    "STRIDE" : 30,
    "N_MC_SAMPLES" : 1,                     # ~~~ relevant for droupout
    #
    # ~~~ For visualization
    "MAKE_GIF" : True,
    "HOW_OFTEN" : 10,                       # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "INITIAL_FRAME_REPETITIONS" : 24,       # ~~~ for how many frames should the state of initialization be rendered
    "FINAL_FRAME_REPETITIONS" : 48,         # ~~~ for how many frames should the state after training be rendered
    "HOW_MANY_INDIVIDUAL_PREDICTIONS" : 6,  # ~~~ how many posterior predictive samples to plot
    "VISUALIZE_DISTRIBUTION_USING_QUANTILES" : True, # ~~~ for dropout, if False, use mean +/- two standard deviatiations; if True, use empirical median and 95% quantile
    "N_POSTERIOR_SAMPLES" : 100,            # ~~~ for dropout, how many samples to use to make the empirical distributions for plotting
    #
    # ~~~ For metrics and visualization
    "N_POSTERIOR_SAMPLES_EVALUATION" : 1000,
    "SHOW_DIAGNOSTICS" : True,
    "SHOW_PLOT" : True
}

#
# ~~~ Define the variable `input_json_filename`
if hasattr(sys,"ps1"):
    #
    # ~~~ If this is an interactive (not srcipted) session, i.e., we are directly typing/pasting in the commands (I do this for debugging), then use the demo json name
    input_json_filename = "demo_nn.json"
    model_save_dir = None
    final_test = False
    overwrite_json = False
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
torch.manual_seed(SEED)

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
DTYPE = getattr(torch,DTYPE)            # ~~~ e.g., DTYPE=="float" (str) -> DTYPE==torch.float (torch.dtype) 
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
except:
    pass

#
# ~~~ Load the network architecture
try:
    model = import_module(f"bnns.models.{MODEL}")   # ~~~ this is equivalent to `import bnns.models.<MODEL> as model`
except:
    model = import_module(MODEL)                    # ~~~ this is equivalent to `import <MODEL> as model` (works if MODEL.py is in the cwd or anywhere on the path)

NN = model.NN.to( device=DEVICE, dtype=DTYPE )


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
optimizer = Optimizer( NN.parameters(), lr=LR )
dataloader = torch.utils.data.DataLoader( D_train, batch_size=BATCH_SIZE )
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
        plot_predictions = plot_bnn_empirical_quantiles if VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
        def plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, nn, extra_std=0., HOW_MANY_INDIVIDUAL_PREDICTIONS=HOW_MANY_INDIVIDUAL_PREDICTIONS, N_POSTERIOR_SAMPLES=N_POSTERIOR_SAMPLES, title=description_of_the_experiment ):
            #
            # ~~~ Draw from the predictive distribuion
            with torch.no_grad():
                predictions = torch.stack([ nn(grid) for _ in range(N_POSTERIOR_SAMPLES) ]).squeeze()
            return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, HOW_MANY_INDIVIDUAL_PREDICTIONS, title )
    #
    # ~~~ Plot the state of the model upon its initialization
    if MAKE_GIF:
        gif = GifMaker()      # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        for j in range(INITIAL_FRAME_REPETITIONS):
            gif.capture( clear_frame_upon_capture=(j+1==INITIAL_FRAME_REPETITIONS) )

#
# ~~~ Establish some variables used for training
N_EPOCHS = non_negative_list( N_EPOCHS, integer_only=True ) # ~~~ supports N_EPOCHS to be a list of integers
STRIDE   = non_negative_list(  STRIDE,  integer_only=True ) # ~~~ supports STRIDE to be a list of integers
PATIENCE = non_negative_list( PATIENCE, integer_only=True ) # ~~~ supports PATIENCE to be a list of integers
DELTA    = non_negative_list( DELTA )                       # ~~~ supports DELTA to be a list of integers
assert np.diff(N_EPOCHS+[N_EPOCHS[-1]+1]).min()>0, "The given sequence N_EPOCHS is not strictly increasing."
train_loss_curve = []
val_loss_curve = []
total_iterations = 0
epochs_completed_so_far = 0
target_epochs = N_EPOCHS.pop(0)
starting_time = time()
first_round = True
keep_training = True
if EARLY_STOPPING:
    #
    # ~~~ Define all len(PATIENCE)*len(DELTA)*len(STRIDE) stopping conditions
    stride_patience_and_delta_stopping_conditions = [
            [
                EarlyStopper( patience=patience, delta=delta )
                for delta, patience in product(DELTA,PATIENCE)
            ]
            for _ in STRIDE
        ]

#
# ~~~ Start the training loop
while keep_training:
    with support_for_progress_bars():   # ~~~ with green progress bars
        stopped_early = False
        pbar = tqdm( desc=description_of_the_experiment, total=target_epochs*len(dataloader), initial=epochs_completed_so_far*len(dataloader), ascii=' >=' )
        # ~~~ 
        #
        ### ~~~
        ## ~~~ Main Loop
        ### ~~~
        #
        # ~~~ The actual training logic (totally conventional, hopefully familiar)
        for e in range( target_epochs - epochs_completed_so_far ):
            for X, y in dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                if not dropout:
                    loss = loss_fn(NN(X),y)
                if dropout:
                    #
                    # ~~~ If the network has dropout, optionally, average over multiple samples
                    loss = 0.
                    for _ in range(N_MC_SAMPLES):
                        loss += loss_fn(NN(X),y)/N_MC_SAMPLES
                #
                # ~~~ Do the gradient-based update
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #
                # ~~~ Report a moving average of train_loss as well as val_loss in the progress bar
                if len(train_loss_curve)>0:
                    pbar_info = { "train_loss": f"{ avg(train_loss_curve[-min(STRIDE):]) :<4.4f}" }
                    if len(val_loss_curve)>0:
                        pbar_info["val_loss"] = f"{ avg(val_loss_curve[-min(STRIDE):]) :<4.4f}"
                    pbar.set_postfix(pbar_info)
                _ = pbar.update()
                #
                # ~~~ Every so often, do some additional stuff, too...
                if (pbar.n+1)%HOW_OFTEN==0:
                    #
                    # ~~~ Plotting logic
                    if data_is_univariate and MAKE_GIF:
                        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
                        gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)
                    #
                    # ~~~ Record a little diagnostic info
                    with torch.no_grad():
                        train_loss_curve.append( loss.item() )
                        val_loss_curve.append( loss_fn(NN(x_test),y_test).item() )
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
        epochs_completed_so_far += e        # ~~~ if stopped_early in the middle if the very first epoch, then epochs_completed_so_far remains 0
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
        # ~~~ Compute the posterior predictive distribution on the testing dataset
        predict = lambda points: torch.stack([ NN(points) for _ in range(N_POSTERIOR_SAMPLES_EVALUATION) ]) if dropout else NN(points)
        with torch.no_grad():
            predictions = predict(x_test)
        try:
            interpolary_grid = data.interpolary_grid.to( device=DEVICE, dtype=DTYPE )
            extrapolary_grid = data.extrapolary_grid.to( device=DEVICE, dtype=DTYPE )
            predictions_on_interpolary_grid = predict(interpolary_grid)
            predictions_on_extrapolary_grid = predict(extrapolary_grid)
        except AttributeError:
            my_warn(f"Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data} nor from just {data}. For the best assessment of the quality of the UQ, please define these variables in the data file (no labels necessary)")
        except:
            raise
        #
        # ~~~ Compute the desired metrics
        hyperparameters["total_iter"] = total_iterations/len(dataloader)
        hyperparameters["epochs_completed"] = epochs_completed_so_far
        hyperparameters["compute_time"] = time() - starting_time
        hyperparameters["patience"] = patience
        hyperparameters["delta"] = delta
        hyperparameters["stride"] = stride
        hyperparameters["val_loss_curve"] = val_loss_curve
        hyperparameters["train_loss_curve"] = train_loss_curve
        if dropout:
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
        else:
            hyperparameters["METRIC_rmse"]      =      rmse( NN, x_test, y_test )
            hyperparameters["METRIC_mae"]       =       mae( NN, x_test, y_test )
            hyperparameters["METRIC_max_norm"]  =  max_norm( NN, x_test, y_test )
        #
        # ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data, as well (the actual heatmaps)
        try:
            S = data.s_truncated.to( device=DEVICE, dtype=DTYPE )
            V = data.V_truncated.to( device=DEVICE, dtype=DTYPE )
            Y = data.unprocessed_y_test.to( device=DEVICE, dtype=DTYPE )
            def predict(x):
                predictions = torch.stack([ NN(x) for _ in range(N_POSTERIOR_SAMPLES_EVALUATION) ]) if dropout else NN(x)
                return predictions * S @ V.T
            with torch.no_grad():
                predictions = predict(x_test)
                predictions_on_interpolary_grid = predict(interpolary_grid)
                predictions_on_extrapolary_grid = predict(extrapolary_grid)
            #
            # ~~~ Compute the desired metrics
            if dropout:
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
            else:
                    hyperparameters["METRIC_unprocessed_rmse"]      =      rmse( predict, x_test, Y )
                    hyperparameters["METRIC_unprocessed_mae"]       =       mae( predict, x_test, Y )
                    hyperparameters["METRIC_unprocessed_max_norm"]  =  max_norm( predict, x_test, Y )
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
                        NN.state_dict(),
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

#
# ~~~ Afterwards, develop the .gif or plot the trained model, if applicable
if data_is_univariate:
    if MAKE_GIF:
        for j in range(FINAL_FRAME_REPETITIONS):
            gif.frames.append( gif.frames[-1] )
        gif.develop( destination=description_of_the_experiment, fps=24 )
        plt.close()
    elif SHOW_PLOT:
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        plt.show()

#
# ~~~ Validate implementation of the algorithm on the synthetic dataset "bivar_trivial"
if data.__name__ == "bnns.data.bivar_trivial" and SHOW_PLOT:
    from bnns.data.univar_missing_middle import x_test, y_test
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot( x_test.cpu(), y_test.cpu(), "--", color="green" )
    with torch.no_grad():
        y_pred = NN(data.D_test.X.to( device=DEVICE, dtype=DTYPE )).mean(dim=-1)
    plt.plot( x_test.cpu(), y_pred.cpu(), "-", color="blue" )
    fig.suptitle("If these lines roughly match, then the algorithm is surely working correctly")
    ax.grid()
    fig.tight_layout()
    plt.show()
