
import numpy as np
import torch
from importlib import import_module
from matplotlib import pyplot as plt
import fiona
from bnns.utils.handling import load_trained_model_from_dataframe


from quality_of_life.my_base_utils import my_warn
try:
    from quality_of_life.my_base_utils import buffer
except:
    from quality_of_life.my_visualization_utils import buffer   # ~~~ deprecated
    print("Please update quality_of_life")



### ~~~
## ~~~ Plotting routines
### ~~~

#
# ~~~ Load coastline land coords (Natalie sent me this code, which I just packaged into a function)
def load_coast_coords(coast_shp_path):
    shape = fiona.open(coast_shp_path)
    coast_coords = []
    for i in range(len(shape)):
        c = np.array(shape[i]['geometry']['coordinates'])
        coast_coords.append(c)
    coast_coords = np.vstack(coast_coords)
    return coast_coords

#
# ~~~ Plot a datapoint from (or a prediction of) the SLOSH dataset as a heatmap
def slosh_heatmap( out, inp=None, show=True ):
    #
    # ~~~ Process `out` and `inp`
    convert = lambda V: V.detach().cpu().numpy().squeeze() if isinstance(V,torch.Tensor) else V
    out = convert(out)
    inp = convert(inp)
    assert out.shape==(49719,), "Required argument `out` should have shape (49719,)"
    if inp is not None:
        assert inp.shape==(5,), "Optional argument `inp` should have shape (5,)"
    #
    # ~~~ Create the actual heat map
    from bnns.data.slosh_70_15_15 import coords_np
    x = coords_np[:,0]
    y = coords_np[:,1]
    figure = plt.figure(figsize=(9,7))
    plt.scatter( x, y, c=out, cmap="viridis" )
    plt.colorbar(label="Storm Surge Heights")
    #
    # ~~~ Create a legend with the input values, if any were supplied, using the hack from https://stackoverflow.com/a/45220580
    if inp is not None:
        plt.plot( [], [], " ", label=f"SLR = {inp[0]}" )
        plt.plot( [], [], " ", label=f"heading = {inp[1]}" )
        plt.plot( [], [], " ", label=f"vel = {inp[2]}" )
        plt.plot( [], [], " ", label=f"pmin = {inp[3]}" )
        plt.plot( [], [], " ", label=f"lat = {inp[4]}" )
    #
    # ~~~ Add the coastline, if possible
    try:
        from bnns import __path__
        data_folder = os.path.join( __path__[0], "data" )
        c = load_coast_coords(os.path.join( data_folder, "ne_10m_coastline", "ne_10m_coastline.shp" ))
        coast_x, coast_y = c[:,0], c[:,1]
        plt.plot( coast_x, coast_y, color="black", linewidth=1 ) #,  label="Coastline" )
        plt.xlim(x.min(),x.max())
        plt.ylim(y.min(),y.max())
    except FileNotFoundError:
        my_warn("Could not find `ne_10m_coastline.shp`. In order to plot the coastline, go to https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/ and click the `Download coastline` button. Unzip the folder, and move the unzipped folder called `ne_10m_coastline` into the working directory or (if the working directory is a subdirectory of the `bnns` repo) the folder bnns/bnns/data")
    #
    # ~~~ Finally just label stuff
    if show:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Heightmap in Cape May County, NJ")
        plt.legend(framealpha=0.9)
        plt.tight_layout()
        plt.show()
    else:
        return figure

#
# ~~~ Somewhat general helper routine for making plots
def univar_figure( fig, ax, grid, green_curve, x_train, y_train, model, title=None, blue_curve=None, **kwargs ):
    with torch.no_grad():
        #
        # ~~~ Green curve and green scatterplot of the data
        _, = ax.plot( grid.cpu(), green_curve.cpu(), color="green", label="Ground Truth", linestyle='--', linewidth=.5 )
        _ = ax.scatter( x_train.cpu(), y_train.cpu(),   color="green" )
        #
        # ~~~ Blue curve(s) of the model
        try:
            ax = blue_curve( model, grid, ax, **kwargs )
        except:
            ax = blue_curve( model, grid, ax ) 
        #
        # ~~~ Finish up
        _ = ax.set_ylim(buffer( y_train.cpu().tolist(), multiplier=0.35 ))
        _ = ax.legend()
        _ = ax.grid()
        _ = ax.set_title(title)
        _ = fig.tight_layout()
    return fig, ax

#
# ~~~ Basically just plot a plain old function
def trivial_sampler(f,grid,ax):
    _, = ax.plot( grid.cpu(), f(grid).cpu(), label="Neural Network", linestyle="-", linewidth=.5, color="blue" )
    return ax

#
# ~~~ Just plot a the model as an ordinary function
def plot_nn(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            NN,             # ~~~ anything with a `__call__` method
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = NN,
            title = "Conventional, Deterministic Training",
            blue_curve = trivial_sampler,
            **kwargs
        )

#
# ~~~ Graph the two standard deviations given pre-computed mean and std
def pre_computed_mean_and_std( mean, std, grid, ax, predictions_include_conditional_std, alpha=0.2, **kwargs ):
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Predicted Posterior Mean", linestyle="-", linewidth=0.5, color="blue" )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    lo, hi = mean-2*std, mean+2*std
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if predictions_include_conditional_std else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Just plot a the model as an ordinary function
def plot_gpr(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            mean,           # ~~~ tensor with the same shape as `grid`
            std,            # ~~~ tensor with the same shape as `grid`
            predictions_include_conditional_std,    # ~~~ Boolean
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need are the vectors `mean` and `std`",
            title="Gaussian Process Regression",
            blue_curve = lambda model,grid,ax: pre_computed_mean_and_std(mean,std,grid,ax,predictions_include_conditional_std),
            **kwargs
        )

#
# ~~~ Graph the mean +/- two standard deviations
def two_standard_deviations( predictions, grid, ax, extra_std, alpha=0.2, how_many_individual_predictions=6, **kwargs ):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *row* of `predictions` is a sample from the posterior predictive distribution
    mean = predictions.mean(dim=0)
    std  =  predictions.std(dim=0) + extra_std
    lo, hi = mean-2*std, mean+2*std
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Posterior Predictive Mean", linestyle="-", linewidth=( 1.5 if how_many_individual_predictions>0 else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        n_posterior_samples = predictions.shape[0]
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), label=("A Sampled Network" if j==max(which_NNs) else ""), linestyle="-", linewidth=0.5, color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if extra_std==0. else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Given a matrix of predictions, plot the empirical mean and +/- 2*std bars
def plot_bnn_mean_and_std(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            predictions,    # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
            extra_std,
            how_many_individual_predictions,
            title,
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need is the matrix of predictions",
            title = title,
            blue_curve = lambda model,grid,ax: two_standard_deviations( predictions, grid, ax, extra_std, how_many_individual_predictions=how_many_individual_predictions ),
            **kwargs
        )

#
# ~~~ Graph a symmetric, empirical 95% confidence interval of a model with a median point estimate
def empirical_quantile( predictions, grid, ax, extra_std, alpha=0.2, how_many_individual_predictions=6, **kwargs ):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *row* of `predictions` is a sample from the posterior predictive distribution
    lo,med,hi = ( predictions + extra_std*torch.randn_like(predictions) ).quantile( q=torch.Tensor([0.025,0.5,0.975]).to(predictions.device), dim=0 )
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), med.cpu(), label="Posterior Predictive Median", linestyle="-", linewidth=( 1.5 if how_many_individual_predictions>0 else 1 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        n_posterior_samples = predictions.shape[0]
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), label=("A Sampled Network" if j==max(which_NNs) else ""), linestyle="-", linewidth=0.5, color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "95% Empirical Quantile Interval"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if extra_std==0. else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Given a matrix of predictions, plot the empirical median and symmetric 95% confidence bars
def plot_bnn_empirical_quantiles(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            predictions,    # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
            extra_std,
            how_many_individual_predictions,
            title,
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need is the matrix of predictions",
            title = title,
            blue_curve = lambda model,grid,ax: empirical_quantile( predictions, grid, ax, extra_std, how_many_individual_predictions=how_many_individual_predictions ),
            **kwargs
        )

#
# ~~~ Load a trained model, based on the dataframe of results you get from hyperparameter search, and then plot it
def plot_trained_model_from_dataframe(
            dataframe,
            i,
            n_samples = 50,
            show = True,
            extra_std = False,
            title = None,
            **other_kwargs
        ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    grid        =  data.x_test.cpu()
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    model = load_trained_model_from_dataframe(dataframe,i)
    if show:
        plot_predictions = plot_bnn_empirical_quantiles if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
        with torch.no_grad():
            try: predictions = model(grid,n=n_samples).squeeze()
            except TypeError: predictions = model(grid).squeeze()
            fig, ax = plt.subplots(figsize=(12,6))
            fig, ax = plot_predictions(
                fig = fig,
                ax = ax,
                grid = grid,
                green_curve = green_curve,
                x_train = x_train_cpu,
                y_train = y_train_cpu,
                predictions = predictions,
                how_many_individual_predictions = dataframe.iloc[i].HOW_MANY_INDIVIDUAL_PREDICTIONS,
                extra_std = extra_std,
                title = title or f"Trained Model i={i}/{len(dataframe)}",
                **other_kwargs
                )
            plt.show()
    return model, x_train_cpu, y_train_cpu, grid, green_curve
