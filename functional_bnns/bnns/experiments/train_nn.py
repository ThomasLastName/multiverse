
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
import os

#
# ~~~ Package-specific utils
from bnns.utils import plot_nn, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, generate_json_filename, set_Dataset_attributes, convert_to_list_and_check_items, non_negative_list, EarlyStopper, add_dropout_to_sequential_relu_network, parse
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
# ~~~ Use argparse to extract the file name `my_hyperparmeters.json` from `python train_nn.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
input_json_filename, model_save_dir, final_test, overwrite_json = parse(hint="try `python train_bnn.py --json demo_bnn`")
hpars = json_to_dict(input_json_filename)

#
# ~~~ Might as well fix a seed, e.g., for randomly shuffling the order of batches during training
torch.manual_seed(hpars["SEED"])

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
DTYPE = getattr(torch,hpars["DTYPE"])           # ~~~ e.g., DTYPE=="float" (str) -> DTYPE==torch.float (torch.dtype) 
torch.set_default_dtype(DTYPE)
Optimizer = getattr(optim,hpars["OPTIMIZER"])   # ~~~ e.g., OPTIMIZER=="Adam" (str) -> Optimizer==optim.Adam

#
# ~~~ Load the data
try:    data = import_module(f'bnns.data.{hpars["DATA"]}')  # ~~~ this is equivalent to `import bnns.data.<hpars["DATA"]> as data`
except: data = import_module(hpars["DATA"])                 # ~~~ this is equivalent to `import <hpars["DATA"]> as data` (works if <hpars["DATA"]>.py is in the cwd or anywhere on the path)

D_train = set_Dataset_attributes( data.D_train, device=hpars["DEVICE"], dtype=DTYPE )
D_test  =  set_Dataset_attributes( data.D_test, device=hpars["DEVICE"], dtype=DTYPE )
D_val   =   set_Dataset_attributes( data.D_val, device=hpars["DEVICE"], dtype=DTYPE ) # ~~~ for hyperparameter evaulation and such, use the validation set instead of the "true" test set
data_is_univariate = (D_train[0][0].numel()==1)
x_train, y_train   =   convert_Dataset_to_Tensors(D_train)
x_test,  y_test    =   convert_Dataset_to_Tensors(D_test if final_test else D_val)

try: grid = data.grid.to( device=hpars["DEVICE"], dtype=DTYPE )
except AttributeError: pass
except: raise

#
# ~~~ Load the network architecture
try:    model = import_module(f'bnns.models.{hpars["ARCHITECTURE"]}')   # ~~~ this is equivalent to `import bnns.models.<hpars["ARCHITECTURE"]> as model`
except: model = import_module(hpars["ARCHITECTURE"])                    # ~~~ this is equivalent to `import <hpars["ARCHITECTURE"]> as model` (works if <hpars["ARCHITECTURE"]>.py is in the cwd or anywhere on the path)
NN = model.NN.to( device=hpars["DEVICE"], dtype=DTYPE )

#
# ~~~ Infer whether or not the model's forward pass is stochastic (e.g., whether or not it's using dropout)
X,_ = next(iter(torch.utils.data.DataLoader( D_train, batch_size=10 )))
with torch.no_grad():
    difference = NN(X)-NN(X)
    dropout = (difference.abs().mean()>1e-6).item()

#
# ~~~ If the model's forward pass is deterministic, but if dropout is desired, then add it now
if (not dropout) and (hpars["DROPOUT"] is not None):
    NN = add_dropout_to_sequential_relu_network( NN, p=hpars["DROPOUT"] )
    dropout = True



### ~~~
## ~~~ Train a conventional neural network
### ~~~

#
# ~~~ The optimizer, dataloader, and loss function
dataloader = torch.utils.data.DataLoader( D_train, batch_size=hpars["BATCH_SIZE"] )
testloader = torch.utils.data.DataLoader( (D_test if final_test else D_val), batch_size=hpars["BATCH_SIZE"] )
optimizer = Optimizer( NN.parameters(), lr=hpars["LR"] )
loss_fn = nn.MSELoss()
n_test_batches = len(testloader)

#
# ~~~ Some naming stuff
description_of_the_experiment = "Conventional, Deterministic Training" if not dropout else "Conventional Training of a Neural Network with Dropout"

#
# ~~~ Use the description_of_the_experiment as the title if no hpars["TITLE"] is specified
try:
    title = description_of_the_experiment if (hpars["TITLE"] is None) else hpars["TITLE"]
except NameError:
    title = description_of_the_experiment

#
# ~~~ Some plotting stuff
if data_is_univariate:
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    if dropout:
        #
        # ~~~ Override the plotting routine `plot_nn` by defining instead a routine which 
        plot_predictions = plot_bnn_empirical_quantiles if hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"] else plot_bnn_mean_and_std
        def plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, nn, extra_std=0., how_many_individual_predictions=hpars["HOW_MANY_INDIVIDUAL_PREDICTIONS"], n_posterior_samples=hpars["N_POSTERIOR_SAMPLES"], title=description_of_the_experiment ):
            #
            # ~~~ Draw from the predictive distribuion
            with torch.no_grad():
                predictions = torch.stack([ nn(grid) for _ in range(n_posterior_samples) ]).squeeze()
            return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, extra_std, how_many_individual_predictions, title )
    #
    # ~~~ Plot the state of the model upon its initialization
    if hpars["MAKE_GIF"]:
        #
        # ~~~ Make the gif, and save `hpars["INITIAL_FRAME_REPETITIONS"]` copies of an identical image of the initial distribution
        gif = GifMaker(title)   # ~~~ essentially just a list of images
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        for j in range(hpars["INITIAL_FRAME_REPETITIONS"]):
            gif.capture( clear_frame_upon_capture=(j+1==hpars["INITIAL_FRAME_REPETITIONS"]) )

#
# ~~~ Establish some variables used for training
N_EPOCHS = non_negative_list( hpars["N_EPOCHS"], integer_only=True )    # ~~~ supports hpars["N_EPOCHS"] to be a list of integers
STRIDE   = non_negative_list(  hpars["STRIDE"],  integer_only=True )    # ~~~ supports hpars["STRIDE"] to be a list of integers
assert np.diff(N_EPOCHS+[N_EPOCHS[-1]+1]).min()>0, "The given sequence N_EPOCHS is not strictly increasing."
train_loss_curve = []
val_loss_curve = []
iter_count = []
epochs_completed_so_far = 0
target_epochs = N_EPOCHS.pop(0)
starting_time = time()
first_round = True
keep_training = True
min_val_loss = float("inf")
if hpars["EARLY_STOPPING"]:
    #
    # ~~~ Define all len(PATIENCE)*len(DELTA)*len(STRIDE) stopping conditions
    PATIENCE = non_negative_list( hpars["PATIENCE"], integer_only=True )        # ~~~ supports hpars["PATIENCE"] to be a list of integers
    DELTA    = convert_to_list_and_check_items( hpars["DELTA"], classes=float ) # ~~~ supports hpars["DELTA"] to be a list of integers
    stride_patience_and_delta_stopping_conditions = [
            [
                EarlyStopper( patience=patience, delta=delta )
                for delta, patience in product(DELTA,PATIENCE)
            ]
            for _ in STRIDE
        ]

#
# ~~~ Do the actual training loop
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
                X, y = X.to(hpars["DEVICE"]), y.to(hpars["DEVICE"])
                #
                # ~~~ Compute the gradient of the loss function on the batch (X,y)
                if not dropout:
                    loss = loss_fn(NN(X),y)
                if dropout: # ~~~ if the network has dropout, optionally average the loss over multiple samples
                    loss = 0.
                    for _ in range(hpars["N_MC_SAMPLES"]):
                        loss += loss_fn(NN(X),y)/hpars["N_MC_SAMPLES"]
                loss.backward()
                #
                # ~~~ Perform the gradient-based update
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
                if (pbar.n+1)%hpars["HOW_OFTEN"]==0:
                    #
                    # ~~~ Plotting logic
                    if data_is_univariate and hpars["MAKE_GIF"]:
                        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
                        gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)
                    #
                    # ~~~ Record a little diagnostic info
                    with torch.no_grad():
                        #
                        # ~~~ Misc.
                        iter_count.append(pbar.n)
                        #
                        # ~~~ Diagnostic info specific to the last seen batch of training data
                        train_loss_curve.append( loss.item() )
                        #
                        # ~~~ Diagnostic info specific to a randomly chosen batch of validation data
                        this_one = np.random.randint(n_test_batches)
                        for b, (X,y) in enumerate(testloader):
                            X, y = X.to(hpars["DEVICE"]), y.to(hpars["DEVICE"])
                            if b==this_one:
                                val_loss = loss_fn(NN(x_test),y_test).item()
                                val_loss_curve.append(val_loss)
                                break
                        #
                        # ~~~ Save only the "best" parameters thus far
                        if val_loss < min_val_loss:
                            best_pars_so_far = NN.state_dict()
                            best_iter_so_far = pbar.n
                            min_val_loss = val_loss
                    #
                    # ~~~ Assess whether or not any new stopping condition is triggered (although, training won't stop until *every* stopping condition is triggered)
                    if hpars["EARLY_STOPPING"]:
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
        # ~~~ Define the predictive process
        def predict(points):
            with torch.no_grad():
                return torch.stack([ NN(points) for _ in range(hpars["N_POSTERIOR_SAMPLES_EVALUATION"]) ]) if dropout else NN(points)
        #
        # ~~~ Compute the posterior predictive distribution on the testing dataset(s)
        predictions = predict(x_test)
        try:
            interpolary_grid = data.interpolary_grid.to( device=hpars["DEVICE"], dtype=DTYPE )
            extrapolary_grid = data.extrapolary_grid.to( device=hpars["DEVICE"], dtype=DTYPE )
            predictions_on_interpolary_grid = predict(interpolary_grid)
            predictions_on_extrapolary_grid = predict(extrapolary_grid)
        except AttributeError:
            my_warn(f"Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data} nor from just {data}. For the best assessment of the quality of the UQ, please define these variables in the data file (no labels necessary)")
        except:
            raise
        #
        # ~~~ Compute the desired metrics
        hpars["total_iter"] = total_iterations/len(dataloader)
        hpars["best_iter"] = best_iter_so_far
        hpars["epochs_completed"] = epochs_completed_so_far
        hpars["compute_time"] = time() - starting_time
        hpars["patience"] = patience
        hpars["delta"] = delta
        hpars["stride"] = stride
        hpars["val_loss_curve"] = val_loss_curve
        hpars["train_loss_curve"] = train_loss_curve
        if dropout:
            hpars["METRIC_rmse_of_median"]             =      rmse_of_median( predictions, y_test )
            hpars["METRIC_rmse_of_mean"]               =        rmse_of_mean( predictions, y_test )
            hpars["METRIC_mae_of_median"]              =       mae_of_median( predictions, y_test )
            hpars["METRIC_mae_of_mean"]                =         mae_of_mean( predictions, y_test )
            hpars["METRIC_max_norm_of_median"]         =  max_norm_of_median( predictions, y_test )
            hpars["METRIC_max_norm_of_mean"]           =    max_norm_of_mean( predictions, y_test )
            hpars["METRIC_median_energy_score"]        =       energy_scores( predictions, y_test ).median().item()
            hpars["METRIC_coverage"]                   =   aggregate_covarge( predictions, y_test, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"] )
            hpars["METRIC_median_avg_inverval_score"]  =  avg_interval_score_of_response_features( predictions, y_test, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"] ).median().item()
            for use_quantiles in (True,False):
                show = hpars["SHOW_DIAGNOSTICS"] and (use_quantiles==hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"])  # ~~~ i.e., diagnostics are requesed, the prediction type mathces the uncertainty type (mean and std. dev., or median and iqr)
                tag = "quantile" if use_quantiles else "pm2_std"
                hpars[f"METRIC_uncertainty_vs_accuracy_slope_{tag}"], hpars[f"METRIC_uncertainty_vs_accuracy_cor_{tag}"]  =  uncertainty_vs_accuracy( predictions, y_test, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"], quantile_accuracy=use_quantiles, show=show, verbose=hpars["SHOW_DIAGNOSTICS"] )
                try:
                    hpars[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{tag}"], hpars[f"METRIC_uncertainty_vs_proximity_cor_{tag}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, use_quantiles, extrapolary_grid, x_train, show=show, title="Uncertainty vs Proximity to Data Outside the Region of Interpolation", verbose=hpars["SHOW_DIAGNOSTICS"] )
                    hpars[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{tag}"], hpars[f"METRIC_uncertainty_vs_proximity_cor_{tag}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, use_quantiles, interpolary_grid, x_train, show=show, title="Uncertainty vs Proximity to Data Within the Region of Interpolation", verbose=hpars["SHOW_DIAGNOSTICS"] )
                    hpars[f"METRIC_extrapolation_uncertainty_spread_{tag}"]  =  uncertainty_spread( predictions_on_extrapolary_grid, use_quantiles )
                    hpars[f"METRIC_interpolation_uncertainty_spread_{tag}"]  =  uncertainty_spread( predictions_on_interpolary_grid, use_quantiles )
                except NameError:
                    pass    # ~~~ the user was already warned "Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}."
                except:
                    raise
        else:
            hpars["METRIC_rmse"]      =      rmse( NN, x_test, y_test )
            hpars["METRIC_mae"]       =       mae( NN, x_test, y_test )
            hpars["METRIC_max_norm"]  =  max_norm( NN, x_test, y_test )
        #
        # ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data, as well (the actual heatmaps)
        try:
            S = data.s_truncated.to( device=hpars["DEVICE"], dtype=DTYPE )
            V = data.V_truncated.to( device=hpars["DEVICE"], dtype=DTYPE )
            Y = data.unprocessed_y_test.to( device=hpars["DEVICE"], dtype=DTYPE )
            def predict(x):
                with torch.no_grad():
                    predictions = torch.stack([ NN(x) for _ in range(hpars["N_POSTERIOR_SAMPLES_EVALUATION"]) ]) if dropout else NN(x)
                    return predictions * S @ V.T
            predictions = predict(x_test)
            predictions_on_interpolary_grid = predict(interpolary_grid)
            predictions_on_extrapolary_grid = predict(extrapolary_grid)
            #
            # ~~~ Compute the desired metrics
            if dropout:
                hpars["METRIC_unprocessed_rmse_of_mean"]                =        rmse_of_mean( predictions, Y )
                hpars["METRIC_unprocessed_rmse_of_median"]              =      rmse_of_median( predictions, Y )
                hpars["METRIC_unprocessed_mae_of_mean"]                 =         mae_of_mean( predictions, Y )
                hpars["METRIC_unprocessed_mae_of_median"]               =       mae_of_median( predictions, Y )
                hpars["METRIC_unprocessed_max_norm_of_mean"]            =    max_norm_of_mean( predictions, Y )
                hpars["METRIC_unprocessed_max_norm_of_median"]          =  max_norm_of_median( predictions, Y )
                hpars["METRIC_unproccessed_coverage"]                   =   aggregate_covarge( predictions, Y, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"] )
                hpars["METRIC_unproccessed_median_energy_score"]        =       energy_scores( predictions, Y ).median().item()
                hpars["METRIC_unproccessed_median_avg_inverval_score"]  =       avg_interval_score_of_response_features( predictions, Y, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"] ).median().item()
                for estimator in ("mean","median"):
                    hpars[f"METRIC_unprocessed_extrapolation_uncertainty_vs_proximity_slope_{estimator}"], hpars[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_extrapolary_grid, (estimator=="median"), extrapolary_grid, x_train, show=hpars["SHOW_DIAGNOSTICS"], title="Uncertainty vs Proximity to Data Outside the Region of Interpolation" )
                    hpars[f"METRIC_unprocessed_interpolation_uncertainty_vs_proximity_slope_{estimator}"], hpars[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"]  =  uncertainty_vs_proximity( predictions_on_interpolary_grid, (estimator=="median"), interpolary_grid, x_train, show=hpars["SHOW_DIAGNOSTICS"], title="Uncertainty vs Proximity to Data Within the Region of Interpolation" )
                    hpars[f"METRIC_unprocessed_uncertainty_vs_accuracy_slope_{estimator}"], hpars[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"]                  =   uncertainty_vs_accuracy( predictions, Y, quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"], quantile_accuracy=(estimator=="median"), show=hpars["SHOW_DIAGNOSTICS"] )
            else:
                    hpars["METRIC_unprocessed_rmse"]      =      rmse( predict, x_test, Y )
                    hpars["METRIC_unprocessed_mae"]       =       mae( predict, x_test, Y )
                    hpars["METRIC_unprocessed_max_norm"]  =  max_norm( predict, x_test, Y )
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
            hpars["filename"] = output_json_filename
            #
            # ~~~ Ok, now actually save the results
            if model_save_dir is not None:
                state_dict_path = os.path.join(
                        model_save_dir,
                        os.path.split(output_json_filename.strip(".json"))[1] + ".pth"
                    )
                hpars["STATE_DICT_PATH"] = state_dict_path
                torch.save(
                        best_pars_so_far,
                        state_dict_path
                    )
            dict_to_json(
                    hpars,
                    output_json_filename,
                    verbose = hpars["SHOW_DIAGNOSTICS"]
                )
        #
        # ~~~ Display the results
        if hpars["SHOW_DIAGNOSTICS"]:
            print_dict(hpars)
        if hpars["SHOW_PLOT"] and keep_training and (not hpars["MAKE_GIF"]):
            fig,ax = plt.subplots(figsize=(12,6))
            fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
            plt.show()

#
# ~~~ Afterwards, develop the .gif or plot the trained model, if applicable
if data_is_univariate:
    if hpars["MAKE_GIF"]:
        for j in range(hpars["FINAL_FRAME_REPETITIONS"]):
            gif.frames.append( gif.frames[-1] )
        gif.develop(fps=24)
        plt.close()
    elif hpars["SHOW_PLOT"]:
        fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN )
        plt.show()

#
# ~~~ Validate implementation of the algorithm on the synthetic dataset "bivar_trivial"
if data.__name__ == "bnns.data.bivar_trivial" and hpars["SHOW_PLOT"]:
    from bnns.data.univar_missing_middle import x_test, y_test
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot( x_test.cpu(), y_test.cpu(), "--", color="green" )
    with torch.no_grad():
        y_pred = NN(data.D_test.X.to( device=hpars["DEVICE"], dtype=DTYPE )).mean(dim=-1)
    plt.plot( x_test.cpu(), y_pred.cpu(), "-", color="blue" )
    fig.suptitle("If these lines roughly match, then the algorithm is surely working correctly")
    ax.grid()
    fig.tight_layout()
    plt.show()

def plot( key, w=None, title=None ):
    lst = hpars[key]
    if w is None: w=30 if len(lst)>50 else 2
    assert len(lst)==len(iter_count)
    plt.plot( moving_average(iter_count,w), moving_average(lst,w) )
    plt.xlabel("Number of Iterations of Gradient Descent")
    plt.ylabel(key)
    plt.suptitle( f"{key} vs. #iter" if title is None else title )
    plt.grid()
    plt.tight_layout()
    plt.show()

if hpars["SHOW_DIAGNOSTICS"]:
    plot( "train_loss_curve", title="Training Loss as Training Progresses" )
    plot( "val_loss_curve", title="Validation Loss as Training Progresses" )
