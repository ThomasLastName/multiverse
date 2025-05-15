
import math
import numpy as np
import pandas as pd
from scipy.integrate import quad
import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain     # ~~~ used to define the prior distribution on network weights


import os
import pytz
from tqdm import tqdm
from glob import glob
from datetime import datetime
from importlib import import_module
from matplotlib import pyplot as plt
import fiona
from quality_of_life.ansi import bcolors
from quality_of_life.my_base_utils import process_for_saving, my_warn, json_to_dict, support_for_progress_bars
try:
    from quality_of_life.my_base_utils import buffer
except:
    from quality_of_life.my_visualization_utils import buffer   # ~~~ deprecated
    print("Please update quality_of_life")



### ~~~
## ~~~ Math stuff
### ~~~

#
# ~~~ Propose a good "prior" standard deviation for a parameter group
def std_per_param(p):
    if len(p.shape)==2:
        #
        # ~~~ For weight matrices, use the standard deviation of pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
        gain = calculate_gain("relu")
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    elif len(p.shape)==1:
        #
        # ~~~ For bias vectors, just use variance==1/len(p) because `_calculate_fan_in_and_fan_out` throws a ValueError(""Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"")
        numb_pars = len(p)
        std = 1/math.sqrt(numb_pars)
    return torch.tensor( std, device=p.device, dtype=p.dtype )

#
# ~~~ Propose good a "prior" standard deviation for weights and biases of a linear layer; mimics pytorch's default initialization, but using a normal instead of uniform distribution (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
def std_per_layer(linear_layer):
    assert isinstance(linear_layer,nn.Linear)
    bound = 1 / math.sqrt(linear_layer.weight.size(1))  # ~~~ see the link above (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
    std = bound / math.sqrt(3)  # ~~~ our reference distribution `uniform_(-bound,bound)` from the deafult pytorch weight initialization has standard deviation bound/sqrt(3), the value of which we copy
    return std

#
# ~~~ Define a class that extends a simple density function into a location scale distribution 
class LocationScaleLogDensity:
    #
    # ~~~ Store the standard log density and test that it is, indeed, standard
    def __init__( self, standard_log_density, check_moments=True ):
        self.standard_log_density = standard_log_density
        if check_moments:
            try:
                self.check_mean_zero_unit_variance()
            except:
                my_warn("Unable to verify mean zero and unit variance in the standard log density. To surpress this warning, pass `check_moments=False` in the `__init__` method.")
    #
    # ~~~ Test that the supposedly "standard" log density has mean zero and unit variance
    def check_mean_zero_unit_variance( self, tol=1e-5 ):
        mean, err_mean = quad( lambda z: z*np.exp(self.standard_log_density(z)), -np.inf, np.inf )
        var, err_var = quad( lambda z: z**2*np.exp(self.standard_log_density(z)), -np.inf, np.inf )
        if abs(mean)>tol or abs(var-1)>tol or err_mean>tol or err_var>tol:
            raise RuntimeError(f"The mean is {mean} and the variance is {var} (should be 0 and 1)")
    #
    # ~~~ Evaluate the log density of mu + sigma*z where z is distributed according to self.standard_log_density
    def __call__( self, where, mu, sigma, multivar=True ):
        #
        # ~~~ Verify that `where-mu` will work
        try:
            assert mu.shape==where.shape
        except:
            assert isinstance(mu,(float,int))
        #
        # ~~~ Verify that `(where-mu)/sigma` will work
        try:
            assert isinstance(sigma,(float,int))
            assert sigma>0
            sigma = torch.tensor( sigma, device=where.device, dtype=where.dtype )
        except:
            assert len(sigma.shape)==0 or sigma.shape==mu.shape # ~~~ either scalar, or a matrix of the same shape is `mu` and `where`
            assert (sigma>0).all(), f"Minimum standard deviation {sigma.min()} is not positive."
        #
        # ~~~ Compute the formula
        marginal_log_probs = self.standard_log_density( (where-mu)/sigma ) - torch.log(sigma)
        return marginal_log_probs.sum() if multivar else marginal_log_probs

#
# ~~~ Compute the log pdf of a multivariate normal distribution with independent coordinates
log_gaussian_pdf = LocationScaleLogDensity( lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) ) )

#
# ~~~ Compute the (appropriately shaped) Jacobian of the final layer of a nerural net (I came up with the formula for the Jacobian, and chat-gpt came up with the generalized vectorized pytorch implementation)
def manual_Jacobian( inputs_to_the_final_layer, number_of_output_features, bias=False ):
    V = inputs_to_the_final_layer
    batch_size, width_of_the_final_layer = V.shape
    total_number_of_predictions = batch_size * number_of_output_features
    I = torch.eye( number_of_output_features, dtype=V.dtype, device=V.device)
    tiled_I = I.repeat( batch_size, 1 )
    tiled_V = V.repeat_interleave( number_of_output_features, dim=0 )
    Jac_wrt_weights = ( tiled_I.unsqueeze(-1) * tiled_V.unsqueeze(1) ).view( total_number_of_predictions, -1 )
    Jac_wrt_biases  =  torch.tile( torch.eye(number_of_output_features), (batch_size,1) ).to( device=Jac_wrt_weights.device, dtype=Jac_wrt_weights.dtype )
    return Jac_wrt_weights if not bias else torch.column_stack([ Jac_wrt_weights, Jac_wrt_biases ])

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
def process_grid_of_unit_cube( grid_of_unit_cube, bounds, extrapolation_percent=0.05, split=True ):
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
    return (extrapolary_grid, interpolary_grid) if split else grid

#
# ~~~ Apply the exact formula for KL( N(mu_0,diag(sigma_0**2)) || N(mu_1,diag(sigma_1**2)) ) (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)
def diagonal_gaussian_kl( mu_0, sigma_0, mu_1, sigma_1 ):
    assert mu_0.shape == mu_1.shape == sigma_0.shape == sigma_1.shape, "Shape assumptions violated."
    assert sigma_0.abs().min()>0 and sigma_1.abs().min()>0, "Variance must be positive."
    return (1/2)*(
            ((sigma_0/sigma_1)**2).sum()                # ~~~ the diagonal case of "tr(Sigma_1^{-1}Sigma_0)"
            - mu_0.numel()                              # ~~~ what wikipedia calls "k"
            + (((mu_1-mu_0)/sigma_1)**2).sum()          # ~~~ the diagonal case of "(mu_0-mu_1)^TSigma_1^{-1}(mu_0-mu_1)" potentially numerically unstble if mu_0\approx\mu_1 and \sigma_1 is small
        ) + sigma_1.log().sum() - sigma_0.log().sum()   # ~~~ the diagonal case of "log(|Sigma_1|/|Sigma_0|)"

#
# ~~~ From a torch.distributions Distribution class, define a method that samples from that standard distribution
class InverseTransformSampler:
    def __init__( self, icdf, generator=None ):
        self.icdf = icdf
        self.generator = generator
    def __call__( self, *shape, device="cpu", dtype=torch.float ):
        U = torch.rand( *shape, generator=self.generator, device=device, dtype=dtype )
        return self.icdf(U)



### ~~~
## ~~~ Non-math non-plotting stuff (e.g., data processing)
### ~~~

#
# ~~~ Flatten and concatenate all the parameters in a model
flatten_parameters = lambda model: torch.cat([ p.view(-1) for p in model.parameters() ])

#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def convert_to_list_and_check_items( x, classes, other_requirement=lambda*args:True ):
    #
    # ~~~ Convert x to a list (if x is already a list, this has no effect)
    try:
        X = list(x)
    except TypeError:
        X = [x]
    except:
        raise
    assert isinstance(X,list), "Unexpected error: both list(x) and [x] failed to create a list out of x."
    #
    # ~~~ Verify the type of each item in the list
    for item in X:
        assert isinstance(item,classes)
        assert other_requirement(item), "The user supplied check was not satisfied."
    #
    # ~~~ Return the list whose items all meet the type and other requirements
    return X

#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def non_negative_list( x, integer_only=False ):
    return convert_to_list_and_check_items(
            x = x,
            classes = int if integer_only else (int,float),
            other_requirement = lambda item: item>=0
        )

#
# ~~~ A standard early stopping rule (https://stackoverflow.com/a/73704579/11595884)
class EarlyStopper:
    def __init__( self, patience=20, delta=0.05 ):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_count = 0
        self.min_val_loss = float('inf')
    def __call__(self,val_loss):
        #
        # ~~~ If the validation loss is decreasing, reset the counter
        if val_loss <= self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        #
        # ~~~ Compute the relative difference between the current `val_loss` and smallest yet `self.min_val_loss`
        if self.min_val_loss>0:
            rel_val_loss = val_loss/self.min_val_loss - 1   # ~~~ `== ( val_loss - self.min_val_loss ) / self.min_val_loss` but more numerically stable
        else:
            rel_val_loss = abs(val_loss-self.min_val_loss) / abs(self.min_val_loss)
        #
        # ~~~ If the current val_loss is "more than delta worse" relative to min_val_loss, increment the counter
        if rel_val_loss > self.delta:
            self.counter += 1
            self.max_count = max( self.counter, self.max_count )
        #
        # ~~~ Stop iff max_count >= patience
        if self.max_count >= self.patience:
            return True
        else:
            return False

#
# ~~~ Load all the .json files in a directory to data frame
def load_filtered_json_files( directory, verbose=True ):
    #
    # ~~~ Load (as a list of dictionaries) all the .json files in a directory that don't start with "RUN_THIS"
    with support_for_progress_bars():
        json_files = glob(os.path.join(directory,'*.json'))
        json_files = [ json for json in json_files if not os.path.split(json)[1].startswith("RUN_THIS") ]
        all_dicts  = [ json_to_dict(json) for json in ( tqdm(json_files,desc="Loading json files") if verbose else json_files ) ]
    #
    # ~~~ Remove from each dictionary any key/value pair where the value is a list, as pandas doesn't like those
    all_filtered_dicts = [
            {
                k:v
                for k,v in dict.items()
                if not isinstance(v,list)
            }
            for dict in all_dicts
        ]
    return pd.DataFrame(all_filtered_dicts)

#
# ~~~ Infer the width of each model
def infer_width_and_depth( dataframe, field="MODEL" ):
    #
    # ~~~ Infer the width of each model
    width_mapping = {}
    for model in dataframe[field].unique():
        text_after_last_underscore = model[model.rfind("_")+1:] # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        width_mapping[model] = int(text_after_last_underscore)
    dataframe["width"] = dataframe[field].map(width_mapping)
    #
    # ~~~ Infer the depth of each model
    depth_mapping = {}
    for model in dataframe[field].unique():
        how_many_underscores = len(model.split("_"))-1 # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        depth_mapping[model] = how_many_underscores-2 if how_many_underscores>1 else 2
    dataframe["depth"] = dataframe[field].map(depth_mapping)
    # print_dict(depth_mapping)
    return dataframe

#
# ~~~ Get the dataframe.iloc[i,"arg"] for all the arguments `args`
def get_attributes_from_row_i(dataframe,i,*args):
    return [
            dataframe.iloc[i][arg]
            for arg in args
        ]

#
# ~~~ Filter a dataframe by multiple attributes
def filter_by_attributes(dataframe,**kwargs):
    filtered_results = dataframe
    for key, value in kwargs.items():
        filtered_results = filtered_results[ filtered_results[key]==value ]
    return filtered_results

#
# ~~~ Load a trained model, based on the string `architecture` that points to the file where the model is defined
def load_trained_model( architecture:str, model:str, state_dict_path ):
    #
    # ~~~ Load the untrained model
    import bnns
    architecture = import_module(f"bnns.models.{architecture}").NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    try:
        model = getattr(bnns,model)(*architecture)
        model.load_state_dict(torch.load(state_dict_path))
        return model
    except:
        architecture.load_state_dict(torch.load(state_dict_path))
        return architecture

#
# ~~~ Load a trained model, based on the dataframe of results you get from hyperparameter search
def load_trained_model_from_dataframe( results_dataframe, i ):
    #
    # ~~~ Load the untrained model
    model = results_dataframe.iloc[i].MODEL
    architecture = results_dataframe.iloc[i].ARCHITECTURE
    state_dict_path = results_dataframe.iloc[i].STATE_DICT_PATH
    return load_trained_model( architecture, model, state_dict_path )

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

#
# ~~~ Add dropout to a standard ReLU network
def add_dropout_to_sequential_relu_network( add_dropout_to_sequential_relu_network, p=0.5 ):
    layers = []
    for layer in add_dropout_to_sequential_relu_network:
        layers.append(layer)
        if isinstance(layer, torch.nn.ReLU):
            layers.append(torch.nn.Dropout(p=p))
    return torch.nn.Sequential(*layers)

#
# ~~~ Generate a list of batch sizes
def get_batch_sizes( N, b ):
    quotient = N // b
    remainder = N % b
    extra = [remainder] if remainder>0 else []
    batch_sizes = [b]*quotient + extra
    assert sum(batch_sizes)==N
    return batch_sizes

def k_smallest_indices( dataframe, column, k ):
    data = dataframe if isinstance(dataframe,pd.Series) else dataframe[column].array
    return np.argpartition(data, k)[:k]

def k_largest_indices( dataframe, column, k ):
    data = dataframe if isinstance(dataframe,pd.Series) else dataframe[column].array
    return np.argpartition(-data, k)[:k]

#
# ~~~ Load a the predictions trained model, based on the dataframe of results you get from hyperparameter search
def get_predictions_and_targets( dataframe, i ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    x_val   = data.x_val.to( device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE )
    targets = data.y_val.to( device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE )
    bnn = load_trained_model_from_dataframe(dataframe,i)
    with torch.no_grad(): predictions = bnn( x_val, n=data.iloc[i].N_POSTERIOR_SAMPLES )
    return predictions, targets

#
# ~~~ Try to get dict[key] but, if that doesn't work, then get source_of_default.default_key instead.
def get_key_or_default( dictionary, key, default ):
    try: return dictionary[key]
    except KeyError:
        my_warn(f'Hyper-parameter "{key}" not specified. Using default value of {default}.')
        return default



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
def plot_trained_model_from_dataframe( dataframe, i, n_samples=200, **kwargs ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    grid        =  data.x_test.cpu()
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    plot_predictions = plot_bnn_empirical_quantiles if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
    bnn = load_trained_model_from_dataframe(dataframe,i)
    with torch.no_grad():
        predictions = bnn(grid,n=n_samples).squeeze()
        fig, ax = plt.subplots(figsize=(12,6))
        fig, ax = plot_predictions(
            fig = fig,
            ax = ax,
            grid = grid,
            green_curve = green_curve,
            x_train = x_train_cpu,
            y_train = y_train_cpu,
            predictions = predictions,
            **kwargs    # ~~~ such as "title" and "extra_std"
            )
        plt.show()
    return bnn
