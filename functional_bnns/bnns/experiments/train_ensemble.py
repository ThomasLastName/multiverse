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
from copy import deepcopy
from time import time
import os

#
# ~~~ Other parts of this package
from bnns.Ensemble import SequentialSteinEnsemble as Ensemble
from bnns.utils.plotting import (
    plot_bnn_mean_and_std,
    plot_bnn_empirical_quantiles,
    GifMaker,
)
from bnns.utils.handling import (
    support_for_progress_bars,
    dict_to_json,
    json_to_dict,
    print_dict,
    my_warn,
    process_for_saving,
    convert_Dataset_to_Tensors,
    set_Dataset_attributes,
    add_dropout_to_sequential_relu_network,
    generate_json_filename,
    convert_to_list_and_check_items,
    non_negative_list,
    EarlyStopper,
    parse,
)
from bnns.utils.math import moving_average
from bnns.metrics import *


### ~~~
## ~~~ Config/setup
### ~~~

#
# ~~~ Use argparse to extract the file name `my_hyperparmeters.json` from `python train_ensemble.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
input_json_filename, model_save_dir, final_test, overwrite_json = parse(
    hint="try `python train_ensemble.py --json demo_ensemble`"
)
hpars = json_to_dict(input_json_filename)

#
# ~~~ Might as well fix a seed, e.g., for randomly shuffling the order of batches during training
torch.manual_seed(hpars["SEED"])

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
DTYPE = getattr(
    torch, hpars["DTYPE"]
)  # ~~~ e.g., DTYPE=="float" (str) -> DTYPE==torch.float (torch.dtype)
torch.set_default_dtype(DTYPE)
Optimizer = getattr(
    optim, hpars["OPTIMIZER"]
)  # ~~~ e.g., OPTIMIZER=="Adam" (str) -> Optimizer==optim.Adam

#
# ~~~ Load the data
try:
    data = import_module(
        f'bnns.data.{hpars["DATA"]}'
    )  # ~~~ this is equivalent to `import bnns.data.<hpars["DATA"]> as data`
except:
    data = import_module(
        hpars["DATA"]
    )  # ~~~ this is equivalent to `import <hpars["DATA"]> as data` (works if <hpars["DATA"]>.py is in the cwd or anywhere on the path)

D_train = set_Dataset_attributes(data.D_train, device=hpars["DEVICE"], dtype=DTYPE)
D_test = set_Dataset_attributes(data.D_test, device=hpars["DEVICE"], dtype=DTYPE)
D_val = set_Dataset_attributes(
    data.D_val, device=hpars["DEVICE"], dtype=DTYPE
)  # ~~~ for hyperparameter evaulation and such, use the validation set instead of the "true" test set
data_is_univariate = D_train[0][0].numel() == 1
x_train, y_train = convert_Dataset_to_Tensors(D_train)
x_test, y_test = convert_Dataset_to_Tensors(D_test if final_test else D_val)


try:
    grid = data.grid.to(device=hpars["DEVICE"], dtype=DTYPE)
except AttributeError:
    pass
except:
    raise

#
# ~~~ Load the network architecture
try:
    model = import_module(
        f'bnns.models.{hpars["ARCHITECTURE"]}'
    )  # ~~~ this is equivalent to `import bnns.models.<hpars["ARCHITECTURE"]> as model`
except:
    model = import_module(
        hpars["ARCHITECTURE"]
    )  # ~~~ this is equivalent to `import <hpars["ARCHITECTURE"]> as model` (works if <hpars["ARCHITECTURE"]>.py is in the cwd or anywhere on the path)
NN = model.NN.to(device=hpars["DEVICE"], dtype=DTYPE)

#
# ~~~ Instantiate an ensemble
ensemble = Ensemble(
    architecture=NN,
    n_copies=hpars["N_MODELS"],
    device=hpars["DEVICE"],
    likelihood_std=torch.tensor(hpars["LIKELIHOOD_STD"]) if hpars["BAYESIAN"] else None,
    bw=hpars["BW"],
)

if hpars["DROPOUT"] is not None:
    assert (
        len(hpars["DROPOUT"]) == hpars["N_MODELS"]
    ), f'If hpars["DROPOUT"] is a list, it must have length equal to hpars["N_MODELS"]={hpars["N_MODELS"]}'
    DROPOUT = convert_to_list_and_check_items(
        hpars["DROPOUT"], classes=(float, int), other_requirement=lambda p: 0 <= p <= 1
    )
    for j in range(hpars["N_MODELS"]):
        ensemble.models[j] = add_dropout_to_sequential_relu_network(
            ensemble.models[j], p=DROPOUT[j]
        )


### ~~~
## ~~~ Train an ensemble, either convenionally, or as the "particles" in SVGD
### ~~~

#
# ~~~ The dataloader and optimizer
dataloader = torch.utils.data.DataLoader(D_train, batch_size=hpars["BATCH_SIZE"])
testloader = torch.utils.data.DataLoader(
    (D_test if final_test else D_val), batch_size=hpars["BATCH_SIZE"]
)

if isinstance(hpars["LR"], list):
    assert (
        len(hpars["LR"]) == hpars["N_MODELS"]
    ), f'If hpars["LR"] is a list, it must have length equal to hpars["N_MODELS"]={hpars["N_MODELS"]}'
    LR = non_negative_list(hpars["LR"])
    param_groups = [
        {"params": model.parameters(), "lr": lr}
        for model, lr in zip(ensemble.models, LR)
    ]
    optimizer = torch.optim.Adam(param_groups)
else:
    optimizer = Optimizer(ensemble.parameters(), lr=hpars["LR"])

n_batches = len(dataloader)
n_test_batches = len(testloader)
n_params = sum(p.numel() for p in ensemble.parameters()) / hpars["N_MODELS"]

#
# ~~~ Some naming stuff
if hpars["BAYESIAN"]:
    if hpars["STEIN"]:
        description_of_the_experiment = "Stein Neural Network Ensemble"
    else:
        description_of_the_experiment = "Ensemble Computation of Posterior Mode"
else:
    if hpars["STEIN"]:
        my_warn(
            'The settings hpars["STEIN"]=True and hpars["BAYESIAN"]=False are incompatible. The former will be ignored.'
        )
    else:
        description_of_the_experiment = "Conventional Neural Network Ensemble"

#
# ~~~ Use the description_of_the_experiment as the title if no hpars["TITLE"] is specified
try:
    title = (
        description_of_the_experiment if (hpars["TITLE"] is None) else hpars["TITLE"]
    )
except NameError:
    title = description_of_the_experiment

#
# ~~~ Some plotting stuff
if data_is_univariate:
    #
    # ~~~ Define some objects used for plotting
    green_curve = data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    #
    # ~~~ Define the main plotting routine
    plot_predictions = (
        plot_bnn_empirical_quantiles
        if hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"]
        else plot_bnn_mean_and_std
    )

    def plot_ensemble(
        fig,
        ax,
        grid,
        green_curve,
        x_train_cpu,
        y_train_cpu,
        ensemble,
        extra_std=(hpars["LIKELIHOOD_STD"] if hpars["EXTRA_STD"] else 0.0),
        how_many_individual_predictions=hpars["HOW_MANY_INDIVIDUAL_PREDICTIONS"],
        title=title,
    ):
        #
        # ~~~ Draw from the posterior predictive distribuion
        with torch.no_grad():
            predictions = ensemble(grid, method="vmap").squeeze()
        return plot_predictions(
            fig,
            ax,
            grid,
            green_curve,
            x_train_cpu,
            y_train_cpu,
            predictions,
            extra_std,
            how_many_individual_predictions,
            title,
        )

    #
    # ~~~ Plot the state of the posterior predictive distribution upon its initialization
    if hpars["MAKE_GIF"]:
        #
        # ~~~ Make the gif, and save `hpars["INITIAL_FRAME_REPETITIONS"]` copies of an identical image of the initial distribution
        gif = GifMaker(title)  # ~~~ essentially just a list of images
        fig, ax = plt.subplots(figsize=(12, 6))
        fig, ax = plot_ensemble(
            fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble
        )
        for j in range(hpars["INITIAL_FRAME_REPETITIONS"]):
            gif.capture(
                clear_frame_upon_capture=(j + 1 == hpars["INITIAL_FRAME_REPETITIONS"])
            )

#
# ~~~ Establish some variables used for training
N_EPOCHS = non_negative_list(
    hpars["N_EPOCHS"], integer_only=True
)  # ~~~ supports hpars["N_EPOCHS"] to be a list of integers
STRIDE = non_negative_list(
    hpars["STRIDE"], integer_only=True
)  # ~~~ supports hpars["STRIDE"] to be a list of integers
assert (
    np.diff(N_EPOCHS + [N_EPOCHS[-1] + 1]).min() > 0
), "The given sequence N_EPOCHS is not strictly increasing."
train_loss_curve = []
val_loss_curve = []
train_lik_curve = []
val_lik_curve = []
log_prior_curve = []
train_acc_curve = []
val_acc_curve = []
iter_count = []
epochs_completed_so_far = 0
best_iter_so_far = 0
target_epochs = N_EPOCHS.pop(0)
starting_time = time()
first_round = True
keep_training = True
min_val_loss = float("inf")
if hpars["EARLY_STOPPING"]:
    #
    # ~~~ Define all len(PATIENCE)*len(DELTA)*len(STRIDE) stopping conditions
    PATIENCE = non_negative_list(
        hpars["PATIENCE"], integer_only=True
    )  # ~~~ supports hpars["PATIENCE"] to be a list of integers
    DELTA = convert_to_list_and_check_items(
        hpars["DELTA"], classes=float
    )  # ~~~ supports hpars["DELTA"] to be a list of integers
    stride_patience_and_delta_stopping_conditions = [
        [
            EarlyStopper(patience=patience, delta=delta)
            for delta, patience in product(DELTA, PATIENCE)
        ]
        for _ in STRIDE
    ]

#
# ~~~ Set "regularization parameters" for a Bayesian loss function (i.e., relative weights of the likelihood and the KL divergence)
if not isinstance(hpars["WEIGHTING"], str):
    my_warn(
        f'Expected hpars["WEIGHTING"] to be a string, but found instead type(hpars["WEIGHTING"])=={type(hpars["WEIGHTING"])}. The loss function will be weighted as if hpars["WEIGHTING"]="standard".'
    )
elif hpars["WEIGHTING"] == "Blundell":
    #
    # ~~~ Follow the suggestion "\pi_i = \frac{2^{M-i}}{2^M-1}" from page 5 of https://arxiv.org/abs/1505.05424
    def decide_weights(**kwargs):
        i = kwargs["b"]
        M = kwargs["n_batches"]
        pi_i = 2 ** (M - (i + 1)) / (
            2**M - 1
        )  # ~~~ note sum( 2**(M-(i+1))/(2**M-1) for i in range(M) )==1
        weight_on_the_kl = pi_i
        weight_on_the_likelihood = 1.0
        return weight_on_the_kl, weight_on_the_likelihood

elif (
    hpars["WEIGHTING"] == "Sun in principle"
):  # (EQUIVALENT TO THE "standard" WEIGHTING BELOW)
    #
    # ~~~ Follow the suggestion "In principle, \lambda should be set as 1/|\mathcal{D}|" in equation (12) of https://arxiv.org/abs/1903.05779
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        D = kwargs["D_train"]
        weight_on_the_kl = 1 / len(D)
        weight_on_the_likelihood = 1 / len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood

elif hpars["WEIGHTING"] == "Sun in practice":
    #
    # ~~~ Follow the suggestion "We used \lambda=1/|\mathcal{D}_s| in practice" in equation (12) of https://arxiv.org/abs/1903.05779
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        weight_on_the_kl = 1 / len(D_s)
        weight_on_the_likelihood = 1 / len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood

elif hpars["WEIGHTING"] == "naive":
    #
    # ~~~ Naively average the marginal KL divergences of each parameter, as well as the marginal likelihoods for each data point
    def decide_weights(**kwargs):
        D_s = kwargs["X"]
        n_params = kwargs["n_params"]
        weight_on_the_kl = 1 / n_params
        weight_on_the_likelihood = 1 / len(D_s)
        return weight_on_the_kl, weight_on_the_likelihood

elif hpars["BAYESIAN"]:
    #
    # ~~~ Downweight the KL divergence in the simplest manner possible to match the expectation of the minibatch estimator of likelihood
    decide_weights = lambda **kwargs: (
        1 / n_batches,
        1.0,
    )  # ~~~ this normalization achchieves an unbiased estimate of the variational loss
    if not hpars["WEIGHTING"] == "standard":
        my_warn(
            f'The given value of hpars["WEIGHTING"] ({hpars["WEIGHTING"]}) was not recognized. Using the default setting of hpars["WEIGHTING"]="standard" instead.'
        )

#
# ~~~ Do the actual training loop
while keep_training:
    with support_for_progress_bars():  # ~~~ this just supports green progress bars
        stopped_early = False
        pbar = tqdm(
            desc=description_of_the_experiment,
            total=target_epochs * len(dataloader),
            initial=epochs_completed_so_far * len(dataloader),
            ascii=" >=",
        )
        # ~~~
        #
        ### ~~~
        ## ~~~ Main Loop
        ### ~~~
        #
        # ~~~ The actual training logic (see train_nn.py for a simpler analogy)
        for e in range(target_epochs - epochs_completed_so_far):
            for b, (X, y) in enumerate(dataloader):
                X, y = X.to(hpars["DEVICE"]), y.to(hpars["DEVICE"])
                #
                # ~~~ Compute the gradient of the loss function on the batch (X,y)
                if hpars["BAYESIAN"]:
                    log_likelihoods = ensemble.log_likelihood_density(X, y)
                    log_priors = ensemble.log_prior_density()
                    beta, alpha = decide_weights(
                        b=b,
                        n_batches=n_test_batches,
                        X=X,
                        D_train=D_train,
                        n_params=n_params,
                    )
                    losses = -(
                        alpha * log_likelihoods + beta * log_priors
                    )  # ~~~ == negative of log posterior density (assuming beta==1==alpha), as if trying to learn the posterior mode
                else:
                    losses = ensemble.mse(X, y)
                losses.sum().backward()  # ~~~ If list==torch.Tensor([ f1(w1), f2(w2), f3(w3) ]), then `list.sum().backward()` is equivalent to calling item.backward() for each item in the list (linearity of the derivative)
                #
                # ~~~ Perform the gradient-based update
                if hpars["BAYESIAN"] and hpars["STEIN"]:
                    ensemble.apply_chain_rule_for_SVGD()
                optimizer.step()
                optimizer.zero_grad()
                #
                # ~~~ Report a moving average of train_loss as well as val_loss in the progress bar
                if len(train_loss_curve) > 0:
                    pbar_info = {
                        "train_loss": f"{ avg(train_loss_curve[-min(STRIDE):]) :<4.4f}"
                    }
                    if len(val_loss_curve) > 0:
                        pbar_info["val_loss"] = (
                            f"{ avg(val_loss_curve[-min(STRIDE):]) :<4.4f}"
                        )
                    if len(val_acc_curve) > 0:
                        pbar_info["val_acc"] = (
                            f"{ avg(val_acc_curve[-min(STRIDE):]) :<4.4f}"
                        )
                    pbar.set_postfix(pbar_info)
                _ = pbar.update()
                #
                # ~~~ Every so often, do some additional stuff, too...
                if (pbar.n + 1) % hpars["HOW_OFTEN"] == 0:
                    #
                    # ~~~ Plotting logic
                    if data_is_univariate and hpars["MAKE_GIF"]:
                        fig, ax = plot_ensemble(
                            fig,
                            ax,
                            grid,
                            green_curve,
                            x_train_cpu,
                            y_train_cpu,
                            ensemble,
                        )
                        gif.capture()
                    #
                    # ~~~ Record a little diagnostic info
                    with torch.no_grad():
                        #
                        # ~~~ Misc.
                        iter_count.append(pbar.n)
                        #
                        # ~~~ Diagnostic info specific to the last seen batch of training data
                        train_loss_curve.append(losses.sum().item())
                        train_acc_curve.append(rmse_of_mean(ensemble(X), y))
                        if hpars["BAYESIAN"]:
                            train_lik_curve.append(log_likelihoods.sum().item())
                            log_prior = log_priors.sum().item()
                            log_prior_curve.append(log_prior)
                        #
                        # ~~~ Diagnostic info specific to a randomly chosen batch of validation data
                        this_one = np.random.randint(n_test_batches)
                        for b, (X, y) in enumerate(testloader):
                            X, y = X.to(hpars["DEVICE"]), y.to(hpars["DEVICE"])
                            if b == this_one:
                                if hpars["BAYESIAN"]:
                                    val_lik = (
                                        ensemble.log_likelihood_density(X, y)
                                        .sum()
                                        .item()
                                    )
                                    val_lik_curve.append(val_lik)
                                    beta, alpha = decide_weights(
                                        b=b,
                                        n_batches=n_test_batches,
                                        X=X,
                                        D_train=D_train,
                                        n_params=n_params,
                                    )
                                    val_loss = -(alpha * val_lik + beta * log_prior)
                                else:
                                    val_loss = ensemble.mse(X, y).sum().item()
                                break
                        val_loss_curve.append(val_loss)
                        val_acc_curve.append(rmse_of_mean(ensemble(X), y))
                        #
                        # ~~~ Save only the "best" parameters thus far
                        if val_loss < min_val_loss:
                            best_pars_so_far = deepcopy(ensemble.state_dict())
                            best_iter_so_far = pbar.n + 1
                            min_val_loss = val_loss
                    #
                    # ~~~ Assess whether or not any new stopping condition is triggered (although, training won't stop until *every* stopping condition is triggered)
                    if hpars["EARLY_STOPPING"]:
                        for i, stride in enumerate(STRIDE):
                            patience_and_delta_stopping_conditions = (
                                stride_patience_and_delta_stopping_conditions[i]
                            )
                            moving_avg_of_val_loss = avg(val_loss_curve[-stride:])
                            for j, early_stopper in enumerate(
                                patience_and_delta_stopping_conditions
                            ):
                                stopped_early = early_stopper(moving_avg_of_val_loss)
                                if stopped_early:
                                    patience, delta = (
                                        early_stopper.patience,
                                        early_stopper.delta,
                                    )
                                    del patience_and_delta_stopping_conditions[j]
                                    if all(
                                        len(lst) == 0
                                        for lst in stride_patience_and_delta_stopping_conditions
                                    ):
                                        keep_training = False
                                    break  # ~~~ break out of the loop over early stoppers
                            if stopped_early:
                                break  # ~~~ break out of the loop over strides
                    if stopped_early:
                        break  # ~~~ break out of the loop over batches
            if stopped_early:
                break  # ~~~ break out of the loop over epochs
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
        def predict(loader):
            with torch.no_grad():
                data_is_unlabeled = isinstance(next(iter(loader)), torch.Tensor)
                predictions = torch.concatenate(
                    [
                        ensemble(batch if data_is_unlabeled else batch[0])
                        for batch in loader
                    ],
                    dim=1,
                )
                if hpars["EXTRA_STD"]:
                    predictions += hpars["LIKELIHOOD_STD"] * torch.randn_like(
                        predictions
                    )
                return predictions

        #
        # ~~~ Compute the posterior predictive distribution on the testing dataset(s)
        predictions = predict(testloader)
        try:
            interpolary_grid = data.interpolary_grid.to(
                device=hpars["DEVICE"], dtype=DTYPE
            )
            extrapolary_grid = data.extrapolary_grid.to(
                device=hpars["DEVICE"], dtype=DTYPE
            )
            batched_interpolary_grid = torch.utils.data.DataLoader(
                interpolary_grid, batch_size=hpars["BATCH_SIZE"]
            )
            batched_extrapolary_grid = torch.utils.data.DataLoader(
                extrapolary_grid, batch_size=hpars["BATCH_SIZE"]
            )
            predictions_on_interpolary_grid = predict(batched_interpolary_grid)
            predictions_on_extrapolary_grid = predict(batched_extrapolary_grid)
        except AttributeError:
            my_warn(
                f"Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}. For the best assessment of the quality of the UQ, please define these variables in the data file (no labels necessary)"
            )
        #
        # ~~~ Compute the desired metrics
        hpars["total_iter"] = total_iterations / len(dataloader)
        hpars["best_iter"] = best_iter_so_far
        hpars["epochs_completed"] = epochs_completed_so_far
        hpars["compute_time"] = time() - starting_time
        hpars["patience"] = patience
        hpars["delta"] = delta
        hpars["stride"] = stride
        hpars["val_loss_curve"] = val_loss_curve
        hpars["train_loss_curve"] = train_loss_curve
        hpars["val_acc_curve"] = val_acc_curve
        hpars["train_acc_curve"] = train_acc_curve
        hpars["val_lik_curve"] = val_lik_curve
        hpars["train_lik_curve"] = train_lik_curve
        hpars["log_prior_curve"] = log_prior_curve
        hpars["train_acc"] = avg(train_loss_curve[-min(STRIDE) :])
        hpars["METRIC_rmse_of_median"] = rmse_of_median(predictions, y_test)
        hpars["METRIC_rmse_of_mean"] = rmse_of_mean(predictions, y_test)
        hpars["METRIC_mae_of_median"] = mae_of_median(predictions, y_test)
        hpars["METRIC_mae_of_mean"] = mae_of_mean(predictions, y_test)
        hpars["METRIC_max_norm_of_median"] = max_norm_of_median(predictions, y_test)
        hpars["METRIC_max_norm_of_mean"] = max_norm_of_mean(predictions, y_test)
        hpars["METRIC_median_energy_score"] = (
            energy_scores(predictions, y_test).median().item()
        )
        hpars["METRIC_coverage"] = aggregate_covarge(
            predictions,
            y_test,
            quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"],
        )
        hpars["METRIC_median_avg_inverval_score"] = (
            avg_interval_score_of_response_features(
                predictions,
                y_test,
                quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"],
            )
            .median()
            .item()
        )
        for use_quantiles in (
            True,
        ):  # ~~~ if (True,False), then the code hangs when hpars["MAKE_GIF"]==True and hpars["DEVICE"]=="cuda"!!! what the f
            show = hpars["SHOW_DIAGNOSTICS"] and (
                use_quantiles == hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"]
            )  # ~~~ i.e., diagnostics are requesed, the prediction type mathces the uncertainty type (mean and std. dev., or median and iqr)
            tag = "quantile" if use_quantiles else "pm2_std"
            (
                hpars[f"METRIC_uncertainty_vs_accuracy_slope_{tag}"],
                hpars[f"METRIC_uncertainty_vs_accuracy_cor_{tag}"],
            ) = uncertainty_vs_accuracy(
                predictions,
                y_test,
                quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"],
                quantile_accuracy=use_quantiles,
                show=show,
                verbose=hpars["SHOW_DIAGNOSTICS"],
            )
            try:
                (
                    hpars[f"METRIC_extrapolation_uncertainty_vs_proximity_slope_{tag}"],
                    hpars[f"METRIC_uncertainty_vs_proximity_cor_{tag}"],
                ) = uncertainty_vs_proximity(
                    predictions_on_extrapolary_grid,
                    use_quantiles,
                    extrapolary_grid,
                    x_train,
                    show=show,
                    title="Uncertainty vs Proximity to Data Outside the Region of Interpolation",
                    verbose=hpars["SHOW_DIAGNOSTICS"],
                )
                (
                    hpars[f"METRIC_interpolation_uncertainty_vs_proximity_slope_{tag}"],
                    hpars[f"METRIC_uncertainty_vs_proximity_cor_{tag}"],
                ) = uncertainty_vs_proximity(
                    predictions_on_interpolary_grid,
                    use_quantiles,
                    interpolary_grid,
                    x_train,
                    show=show,
                    title="Uncertainty vs Proximity to Data Within the Region of Interpolation",
                    verbose=hpars["SHOW_DIAGNOSTICS"],
                )
                hpars[f"METRIC_extrapolation_uncertainty_spread_{tag}"] = (
                    uncertainty_spread(predictions_on_extrapolary_grid, use_quantiles)
                )
                hpars[f"METRIC_interpolation_uncertainty_spread_{tag}"] = (
                    uncertainty_spread(predictions_on_interpolary_grid, use_quantiles)
                )
            except NameError:
                pass  # ~~~ the user was already warned "Could import `extrapolary_grid` or `interpolary_grid` from bnns.data.{data}."
            except:
                raise
        #
        # ~~~ For the SLOSH dataset, run all the same metrics on the unprocessed data (the actual heatmaps)
        try:
            S = data.s_truncated.to(device=hpars["DEVICE"], dtype=DTYPE)
            V = data.V_truncated.to(device=hpars["DEVICE"], dtype=DTYPE)
            Y = data.unprocessed_y_test.to(device=hpars["DEVICE"], dtype=DTYPE)

            def predict(loader):
                with torch.no_grad():
                    data_is_unlabeled = isinstance(next(iter(loader)), torch.Tensor)
                    predictions = torch.concatenate(
                        [
                            ensemble(batch if data_is_unlabeled else batch[0])
                            for batch in loader
                        ],
                        dim=1,
                    )
                    if hpars["EXTRA_STD"]:
                        predictions += hpars["LIKELIHOOD_STD"] * torch.randn_like(
                            predictions
                        )
                    return predictions.mean(dim=0, keepdim=True) * S @ V.T

            predictions = predict(x_test)
            predictions_on_interpolary_grid = predict(batched_interpolary_grid)
            predictions_on_extrapolary_grid = predict(batched_extrapolary_grid)
            #
            # ~~~ Compute the desired metrics
            hpars["METRIC_unprocessed_rmse_of_mean"] = rmse_of_mean(predictions, Y)
            hpars["METRIC_unprocessed_rmse_of_median"] = rmse_of_median(predictions, Y)
            hpars["METRIC_unprocessed_mae_of_mean"] = mae_of_mean(predictions, Y)
            hpars["METRIC_unprocessed_mae_of_median"] = mae_of_median(predictions, Y)
            hpars["METRIC_unprocessed_max_norm_of_mean"] = max_norm_of_mean(
                predictions, Y
            )
            hpars["METRIC_unprocessed_max_norm_of_median"] = max_norm_of_median(
                predictions, Y
            )
            hpars["METRIC_unproccessed_coverage"] = aggregate_covarge(
                predictions,
                Y,
                quantile_uncertainty=hpars["VISUALIZE_DISTRIBUTION_USING_QUANTILES"],
            )
            hpars["METRIC_unproccessed_median_energy_score"] = (
                energy_scores(predictions, Y).median().item()
            )
            hpars["METRIC_unproccessed_median_avg_inverval_score"] = (
                avg_interval_score_of_response_features(
                    predictions,
                    Y,
                    quantile_uncertainty=hpars[
                        "VISUALIZE_DISTRIBUTION_USING_QUANTILES"
                    ],
                )
                .median()
                .item()
            )
            for estimator in ("mean", "median"):
                (
                    hpars[
                        f"METRIC_unprocessed_extrapolation_uncertainty_vs_proximity_slope_{estimator}"
                    ],
                    hpars[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"],
                ) = uncertainty_vs_proximity(
                    predictions_on_extrapolary_grid,
                    (estimator == "median"),
                    extrapolary_grid,
                    x_train,
                    show=hpars["SHOW_DIAGNOSTICS"],
                    title="Uncertainty vs Proximity to Data Outside the Region of Interpolation",
                )
                (
                    hpars[
                        f"METRIC_unprocessed_interpolation_uncertainty_vs_proximity_slope_{estimator}"
                    ],
                    hpars[f"METRIC_uncertainty_vs_proximity_cor_{estimator}"],
                ) = uncertainty_vs_proximity(
                    predictions_on_interpolary_grid,
                    (estimator == "median"),
                    interpolary_grid,
                    x_train,
                    show=hpars["SHOW_DIAGNOSTICS"],
                    title="Uncertainty vs Proximity to Data Within the Region of Interpolation",
                )
                (
                    hpars[
                        f"METRIC_unprocessed_uncertainty_vs_accuracy_slope_{estimator}"
                    ],
                    hpars[f"METRIC_uncertainty_vs_accuracy_cor_{estimator}"],
                ) = uncertainty_vs_accuracy(
                    predictions,
                    Y,
                    quantile_uncertainty=hpars[
                        "VISUALIZE_DISTRIBUTION_USING_QUANTILES"
                    ],
                    quantile_accuracy=(estimator == "median"),
                    show=hpars["SHOW_DIAGNOSTICS"],
                )
        except AttributeError:
            pass
        #
        # ~~~ Save the results
        if input_json_filename.startswith("demo"):
            my_warn(
                f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)'
            )
        else:
            #
            # ~~~ Put together the output json filename
            output_json_filename = (
                input_json_filename if overwrite_json else generate_json_filename()
            )
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
                    os.path.split(output_json_filename.strip(".json"))[1] + ".pth",
                )
                hpars["STATE_DICT_PATH"] = state_dict_path
                torch.save(best_pars_so_far, state_dict_path)
            dict_to_json(hpars, output_json_filename, verbose=hpars["SHOW_DIAGNOSTICS"])
        #
        # ~~~ Display the results
        if hpars["SHOW_DIAGNOSTICS"]:
            print_dict(hpars)
        if hpars["SHOW_PLOT"] and keep_training and (not hpars["MAKE_GIF"]):
            fig, ax = plt.subplots(figsize=(12, 6))
            fig, ax = plot_ensemble(
                fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble
            )
            plt.show()

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if data_is_univariate:
    if hpars["MAKE_GIF"]:
        for j in range(hpars["FINAL_FRAME_REPETITIONS"]):
            gif.frames.append(gif.frames[-1])
        gif.develop(fps=24)
        plt.close()
    elif hpars["SHOW_PLOT"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig, ax = plot_ensemble(
            fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble
        )
        plt.show()


### ~~~
## ~~~ Debugging diagnostics
### ~~~


def plot(key, w=None, title=None):
    lst = hpars[key]
    if w is None:
        w = 30 if len(lst) > 50 else 2
    assert len(lst) == len(iter_count)
    plt.plot(moving_average(iter_count, w), moving_average(lst, w))
    plt.xlabel("Number of Iterations of Gradient Descent")
    plt.ylabel(key)
    plt.suptitle(f"{key} vs. #iter" if title is None else title)
    plt.grid()
    plt.tight_layout()
    plt.show()


if hpars["SHOW_DIAGNOSTICS"]:
    if hpars["BAYESIAN"]:
        plot("log_prior_curve", title="Prior Probabiliy as Training Progresses")
    plot("train_loss_curve", title="Training Loss as Training Progresses")
    plot("val_loss_curve", title="Validation Loss as Training Progresses")
