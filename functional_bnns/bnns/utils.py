
import math
import numpy as np
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain     # ~~~ used to define the prior distribution on network weights

import os
import pytz
from datetime import datetime
from matplotlib import pyplot as plt
import fiona
from quality_of_life.ansi import bcolors
from quality_of_life.my_base_utils import process_for_saving, my_warn
try:
    from quality_of_life.my_base_utils import buffer
except:
    from quality_of_life.my_visualization_utils import buffer   # ~~~ deprecated
    print("Please update quality_of_life")



### ~~~
## ~~~ Math stuff
### ~~~

#
# ~~~ Compute the log pdf of a multivariate normal distribution with independent coordinates
def log_gaussian_pdf( where, mu, sigma ):
    assert mu.shape==where.shape
    try:
        assert len(sigma.shape)==0 or sigma.shape==mu.shape # ~~~ either scalar, or a matrix of the same shape is `mu` and `where`
        assert (sigma>0).all()
    except:
        assert isinstance(1,(float,int))
        assert sigma>0
    marginal_log_probs = -((where-mu)/sigma)**2/2 - torch.log( math.sqrt(2*torch.pi)*sigma )   # ~~~ note: isn't (x-mu)/sigma numerically unstable, like numerical differentiation?
    return marginal_log_probs.sum()

#
# ~~~ Define what we want the prior std. to be for each group of model parameters
def get_std(p):
    if len(p.shape)==1: # ~~~ for the biase vectors, take variance=1/length
        numb_pars = len(p)
        std = 1/math.sqrt(numb_pars)
    else:   # ~~~ for the weight matrices, mimic pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
        gain = calculate_gain("relu")
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return torch.tensor( std, device=p.device, dtype=p.dtype )

#
# ~~~ Compute the (appropriately shaped) Jacobian of the final layer of a nerural net (I came up with the formula for the Jacobian, and chat-gpt came up with the generalized vectorized pytorch implementation)
def manual_Jacobian( inputs_to_the_final_layer, number_of_output_features ):
    V = inputs_to_the_final_layer
    batch_size, width_of_the_final_layer = V.shape
    total_number_of_predictions = batch_size * number_of_output_features
    I = torch.eye( number_of_output_features, dtype=V.dtype, device=V.device)
    tiled_I = I.repeat( batch_size, 1 )
    tiled_V = V.repeat_interleave( number_of_output_features, dim=0 )
    result = tiled_I.unsqueeze(-1) * tiled_V.unsqueeze(1)
    return result.view( total_number_of_predictions, -1 )

#
# ~~~ Compute the slope and intercept in linear regression
def lm(y,x):
    try:
        var = (x**2).mean() - x.mean()**2
        slope = (x*y).mean()/var - x.mean()*y.mean()/var
        intercept = y.mean() - slope*x.mean()
        return slope.item(), intercept.item()
    except:
        var = np.mean(x**2) - np.mean(x)**2
        slope = np.mean(x*y)/var - np.mean(x)*np.mean(y)/var
        intercept = np.mean(y) - slope*np.mean(x)
        return slope, intercept
        

#
# ~~~ Compute the empirical correlation coefficient between two vectors
def cor(u,w):
    try:
        stdstd = ((u**2).mean() - u.mean()**2).sqrt() * ((w**2).mean() - w.mean()**2).sqrt()
        return ((u*w).mean()/stdstd - u.mean()*w.mean()/stdstd).item()
    except:
        return np.corrcoef(u,w)[0,1]

#
# ~~~ Compute an empirical 95% confidence interval
iqr = lambda tensor, dim=None: tensor.quantile( q=torch.Tensor([0.25,0.75]).to(tensor.device), dim=dim ).diff(dim=0).squeeze(dim=0)

#
# ~~~ Do polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    try:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
    except:
        pass
    coeffs = np.polyfit( x, y, deg=degree )
    poly = np.poly1d(coeffs)
    R_squared = cor(poly(x),y)**2
    return poly, coeffs, R_squared

#
# ~~~ Start with a grid of points in the unit cube, and then transform it to the desired bounds, includeing some exaggeration of the bounds
def process_grid_of_unit_cube( grid_of_unit_cube, bounds, extrapolation_percent=0.05 ):
    lo = bounds[:,0].clone()
    hi = bounds[:,1].clone()
    range = hi-lo
    hi += extrapolation_percent*range
    lo -= extrapolation_percent*range
    grid = lo + (hi-lo)*grid_of_unit_cube
    extrapolary_grid = grid[torch.where(torch.logical_or(
            torch.any( grid>bounds[:,1], dim=1 ),
            torch.any( grid<bounds[:,0], dim=1 )
        ))]
    interpolary_grid = grid[torch.where(torch.logical_and(
            torch.all( grid<=bounds[:,1], dim=1 ),
            torch.all( grid>=bounds[:,0], dim=1 )
        ))]
    return extrapolary_grid, interpolary_grid


#
# ~~~ Draw uniform random samples from the convex hull of `points` with shape (n_points,d)
def sample_from_convex_hull( points, n_samples, noise=0.):
    #
    # ~~~ Reshape the points to (n_points,d) in case it's a flat vector, in which case d==1
    n_points,d = points.reshape(points.shape[0],-1).shape
    #
    # ~~~ Sample uniformly at random from the probability simplex (https://mathoverflow.net/questions/368226/uniform-distribution-on-a-simplex)
    exp_dist = torch.distributions.Exponential(rate=1.0) 
    weights = exp_dist.sample(sample_shape=(n_samples,n_points)).to( device=points.device, dtype=points.dtype )
    weights /= weights.sum( dim=1, keepdim=True )    
    #
    # ~~~ Compute convex combinations of the original points using the random weights
    points_from_convex_hull = weights@points
    return points_from_convex_hull + noise*torch.randn_like(points_from_convex_hull)

# points = torch.randn(1000,5)
# centroid = points.mean(dim=0)
# magnitudes = (points**2).sum(dim=1).sqrt()
# normalized_points = (points.T/magnitudes).T
# assert normalized_points.shape==(1000,5) and (normalized_points**2).sum(dim=1).sqrt().allclose(torch.ones(1000))
# samples = sample_from_convex_hull(normalized_points,900)
# print("Average distance from the centroid:", (samples-centroid).norm(dim=1).mean().item())
#
# ~~~ Generate all the weight combinations (n1/res,n2/res,...,nd/res) where n1,...,nd are non-negative integers summing to `res`
# def generate_Barycentric_grid(d,res,weights_only=True,points=None):
#     if not weights_only:
#         n_points, d = points.reshape(points.shape[0],-1).shape
#     grid_of_weights = torch.tensor(list(product( *d*[[j for j in range(res+1)]])))      # ~~~ all combinations (n1,...,nd) where 0<=ni<=res for all i
#     convex_weights = grid_of_weights[torch.where(grid_of_weights.sum(dim=1)==res)]/res  # ~~~ filter for only those which sum to `res`, and normalize to sum to 1
#     assert convex_weights.shape == ( math.comb(res+d-1,res), d) # ~~~ https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)#Theorem_two
#     return convex_weights if weights_only else convex_weights@points



### ~~~
## ~~~ Non-math non-plotting stuff (e.g., data processing)
### ~~~

#
# ~~~ Generate a .json filename based on the current datetime
def generate_json_filename(verbose=True,message=None):
    #
    # ~~~ Generate a .json filename
    time = datetime.now(pytz.timezone('US/Central'))        # ~~~ current date and time CST
    file_name = str(time)
    file_name = file_name[:file_name.find(".")]             # ~~~ remove the number of milliseconds (indicated with ".") 
    file_name = file_name.replace(" ","_").replace(":","-") # ~~~ replace blank space (between date and time) with an underscore and colons (hr:mm:ss) with dashes
    file_name = process_for_saving(file_name+".json")       # ~~~ procsess_for_saving("path_that_exists.json") returns "path_that_exists (1).json"
    #
    # ~~~ Craft a message to print
    if verbose:
        if time.hour > 12:
            hour = time.hour - 12
            suffix = "pm"
        else:
            hour = time.hour
            suffix = "am"
        base_message = bcolors.OKBLUE + f"    Generating file name {file_name} at {hour}:{time.minute:02d}{suffix} CST" + bcolors.HEADER
        if message is not None:
            if not message[0]==" ":
                message = " " + message
            base_message += message
        print(base_message)
    return file_name

#
# ~~~ My version of the missing feature: a `dataset.to` method
def set_Dataset_attributes( dataset, device, dtype ):
    try:
        #
        # ~~~ Directly access and modify the underlying tensors
        dataset.X = dataset.X.to( device=device, dtype=dtype )
        dataset.y = dataset.y.to( device=device, dtype=dtype )
        return dataset
    except AttributeError:
        #
        # ~~~ Redefine the __getattr__ method (this is hacky; I don't know a better way; also, chat-gpt proposed this)
        class ModifiedDataset(torch.utils.data.Dataset):
            def __init__(self,original_dataset):
                self.original_dataset = original_dataset
                self.device = device
                self.dtype = dtype
            def __getitem__(self,index):
                x, y = self.original_dataset[index]
                return x.to( device=self.device, dtype=self.dtype ), y.to( device=self.device, dtype=self.dtype )
            def __len__(self):
                return len(self.original_dataset)
        return ModifiedDataset(dataset)



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
        _ = ax.set_title( description_of_the_experiment if title is None else title )
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
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
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
    _, = ax.plot( grid.cpu(), med.cpu(), label="Posterior Predictive Median", linestyle="-", linewidth=( 1.5 if how_many_individual_predictions>0 else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        n_posterior_samples = predictions.shape[0]
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[j,:].cpu(), linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
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


