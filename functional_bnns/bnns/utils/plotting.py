import numpy as np
import torch
import os
import fiona
from PIL import Image
from io import BytesIO
from importlib import import_module
from matplotlib import pyplot as plt
from bnns.utils.handling import (
    load_trained_model_from_dataframe,
    my_warn,
    process_for_saving,
)


### ~~~
## ~~~ Plotting routines
### ~~~


#
# ~~~ Load coastline land coords (Natalie sent me this code, which I just packaged into a function)
def load_coast_coords(coast_shp_path):
    shape = fiona.open(coast_shp_path)
    coast_coords = []
    for i in range(len(shape)):
        c = np.array(shape[i]["geometry"]["coordinates"])
        coast_coords.append(c)
    coast_coords = np.vstack(coast_coords)
    return coast_coords


#
# ~~~ Plot a datapoint from (or a prediction of) the SLOSH dataset as a heatmap
def slosh_heatmap(out, inp=None, show=True):
    #
    # ~~~ Process `out` and `inp`
    convert = lambda V: (
        V.detach().cpu().numpy().squeeze() if isinstance(V, torch.Tensor) else V
    )
    out = convert(out)
    inp = convert(inp)
    assert out.shape == (49719,), "Required argument `out` should have shape (49719,)"
    if inp is not None:
        assert inp.shape == (5,), "Optional argument `inp` should have shape (5,)"
    #
    # ~~~ Create the actual heat map
    from bnns.data.slosh_70_15_15 import coords_np

    x = coords_np[:, 0]
    y = coords_np[:, 1]
    figure = plt.figure(figsize=(9, 7))
    plt.scatter(x, y, c=out, cmap="viridis")
    plt.colorbar(label="Storm Surge Heights")
    #
    # ~~~ Create a legend with the input values, if any were supplied, using the hack from https://stackoverflow.com/a/45220580
    if inp is not None:
        plt.plot([], [], " ", label=f"SLR = {inp[0]}")
        plt.plot([], [], " ", label=f"heading = {inp[1]}")
        plt.plot([], [], " ", label=f"vel = {inp[2]}")
        plt.plot([], [], " ", label=f"pmin = {inp[3]}")
        plt.plot([], [], " ", label=f"lat = {inp[4]}")
    #
    # ~~~ Add the coastline, if possible
    try:
        from bnns import __path__

        data_folder = os.path.join(__path__[0], "data")
        c = load_coast_coords(
            os.path.join(data_folder, "ne_10m_coastline", "ne_10m_coastline.shp")
        )
        coast_x, coast_y = c[:, 0], c[:, 1]
        plt.plot(coast_x, coast_y, color="black", linewidth=1)  # ,  label="Coastline" )
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
    except FileNotFoundError:
        my_warn(
            "Could not find `ne_10m_coastline.shp`. In order to plot the coastline, go to https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/ and click the `Download coastline` button. Unzip the folder, and move the unzipped folder called `ne_10m_coastline` into the working directory or (if the working directory is a subdirectory of the `bnns` repo) the folder bnns/bnns/data"
        )
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
def univar_figure(
    fig,
    ax,
    grid,
    green_curve,
    x_train,
    y_train,
    model,
    title=None,
    blue_curve=None,
    **kwargs,
):
    with torch.no_grad():
        #
        # ~~~ Green curve and green scatterplot of the data
        (_,) = ax.plot(
            grid.cpu(),
            green_curve.cpu(),
            color="green",
            label="Ground Truth",
            linestyle="--",
            linewidth=0.5,
        )
        _ = ax.scatter(x_train.cpu(), y_train.cpu(), color="green")
        #
        # ~~~ Blue curve(s) of the model
        try:
            ax = blue_curve(model, grid, ax, **kwargs)
        except:
            ax = blue_curve(model, grid, ax)
        #
        # ~~~ Finish up
        _ = ax.set_ylim(buffer(y_train.cpu().tolist(), multiplier=0.35))
        _ = ax.legend()
        _ = ax.grid()
        _ = ax.set_title(title)
        _ = fig.tight_layout()
    return fig, ax


#
# ~~~ Basically just plot a plain old function
def trivial_sampler(f, grid, ax):
    (_,) = ax.plot(
        grid.cpu(),
        f(grid).cpu(),
        label="Neural Network",
        linestyle="-",
        linewidth=0.5,
        color="blue",
    )
    return ax


#
# ~~~ Just plot a the model as an ordinary function
def plot_nn(
    fig,
    ax,
    grid,
    green_curve,  # ~~~ tensor with the same shape as `grid`
    x_train,
    y_train,  # ~~~ tensor with the same shape as `x_train`
    NN,  # ~~~ anything with a `__call__` method
    **kwargs,
):
    return univar_figure(
        fig,
        ax,
        grid,
        green_curve,
        x_train,
        y_train,
        model=NN,
        title="Conventional, Deterministic Training",
        blue_curve=trivial_sampler,
        **kwargs,
    )


#
# ~~~ Graph the two standard deviations given pre-computed mean and std
def pre_computed_mean_and_std(
    mean, std, grid, ax, predictions_include_conditional_std, alpha=0.2, **kwargs
):
    #
    # ~~~ Graph the median as a blue curve
    (_,) = ax.plot(
        grid.cpu(),
        mean.cpu(),
        label="Predicted Posterior Mean",
        linestyle="-",
        linewidth=0.5,
        color="blue",
    )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    lo, hi = mean - 2 * std, mean + 2 * std
    _ = ax.fill_between(
        grid.cpu(),
        lo.cpu(),
        hi.cpu(),
        facecolor="blue",
        interpolate=True,
        alpha=alpha,
        label=(
            tittle
            if predictions_include_conditional_std
            else tittle + " Including Measurment Noise"
        ),
    )
    return ax


#
# ~~~ Just plot a the model as an ordinary function
def plot_gpr(
    fig,
    ax,
    grid,
    green_curve,  # ~~~ tensor with the same shape as `grid`
    x_train,
    y_train,  # ~~~ tensor with the same shape as `x_train`
    mean,  # ~~~ tensor with the same shape as `grid`
    std,  # ~~~ tensor with the same shape as `grid`
    predictions_include_conditional_std,  # ~~~ Boolean
    **kwargs,
):
    return univar_figure(
        fig,
        ax,
        grid,
        green_curve,
        x_train,
        y_train,
        model="None! All we need are the vectors `mean` and `std`",
        title="Gaussian Process Regression",
        blue_curve=lambda model, grid, ax: pre_computed_mean_and_std(
            mean, std, grid, ax, predictions_include_conditional_std
        ),
        **kwargs,
    )


#
# ~~~ Graph the mean +/- two standard deviations
def two_standard_deviations(
    predictions,
    grid,
    ax,
    extra_std,
    alpha=0.2,
    how_many_individual_predictions=6,
    **kwargs,
):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *row* of `predictions` is a sample from the posterior predictive distribution
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0) + extra_std
    lo, hi = mean - 2 * std, mean + 2 * std
    #
    # ~~~ Graph the median as a blue curve
    (_,) = ax.plot(
        grid.cpu(),
        mean.cpu(),
        label="Posterior Predictive Mean",
        linestyle="-",
        linewidth=(1.5 if how_many_individual_predictions > 0 else 0.5),
        color="blue",
    )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions > 0:
        n_posterior_samples = predictions.shape[0]
        which_NNs = (
            np.linspace(
                1,
                n_posterior_samples,
                min(n_posterior_samples, how_many_individual_predictions),
                dtype=np.int32,
            )
            - 1
        ).tolist()
        for j in which_NNs:
            (_,) = ax.plot(
                grid.cpu(),
                predictions[j, :].cpu(),
                label=("A Sampled Network" if j == max(which_NNs) else ""),
                linestyle="-",
                linewidth=0.5,
                color="blue",
                alpha=(alpha + 1) / 2,
            )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    _ = ax.fill_between(
        grid.cpu(),
        lo.cpu(),
        hi.cpu(),
        facecolor="blue",
        interpolate=True,
        alpha=alpha,
        label=(tittle if extra_std == 0.0 else tittle + " Including Measurment Noise"),
    )
    return ax


#
# ~~~ Given a matrix of predictions, plot the empirical mean and +/- 2*std bars
def plot_bnn_mean_and_std(
    fig,
    ax,
    grid,
    green_curve,  # ~~~ tensor with the same shape as `grid`
    x_train,
    y_train,  # ~~~ tensor with the same shape as `x_train`
    predictions,  # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
    extra_std,
    how_many_individual_predictions,
    title,
    **kwargs,
):
    return univar_figure(
        fig,
        ax,
        grid,
        green_curve,
        x_train,
        y_train,
        model="None! All we need is the matrix of predictions",
        title=title,
        blue_curve=lambda model, grid, ax: two_standard_deviations(
            predictions,
            grid,
            ax,
            extra_std,
            how_many_individual_predictions=how_many_individual_predictions,
        ),
        **kwargs,
    )


#
# ~~~ Graph a symmetric, empirical 95% confidence interval of a model with a median point estimate
def empirical_quantile(
    predictions,
    grid,
    ax,
    extra_std,
    alpha=0.2,
    how_many_individual_predictions=6,
    **kwargs,
):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *row* of `predictions` is a sample from the posterior predictive distribution
    lo, med, hi = (predictions + extra_std * torch.randn_like(predictions)).quantile(
        q=torch.Tensor([0.025, 0.5, 0.975]).to(predictions.device), dim=0
    )
    #
    # ~~~ Graph the median as a blue curve
    (_,) = ax.plot(
        grid.cpu(),
        med.cpu(),
        label="Posterior Predictive Median",
        linestyle="-",
        linewidth=(1.5 if how_many_individual_predictions > 0 else 1),
        color="blue",
    )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions > 0:
        n_posterior_samples = predictions.shape[0]
        which_NNs = (
            np.linspace(
                1,
                n_posterior_samples,
                min(n_posterior_samples, how_many_individual_predictions),
                dtype=np.int32,
            )
            - 1
        ).tolist()
        for j in which_NNs:
            (_,) = ax.plot(
                grid.cpu(),
                predictions[j, :].cpu(),
                label=("A Sampled Network" if j == max(which_NNs) else ""),
                linestyle="-",
                linewidth=0.5,
                color="blue",
                alpha=(alpha + 1) / 2,
            )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "95% Empirical Quantile Interval"
    _ = ax.fill_between(
        grid.cpu(),
        lo.cpu(),
        hi.cpu(),
        facecolor="blue",
        interpolate=True,
        alpha=alpha,
        label=(tittle if extra_std == 0.0 else tittle + " Including Measurment Noise"),
    )
    return ax


#
# ~~~ Given a matrix of predictions, plot the empirical median and symmetric 95% confidence bars
def plot_bnn_empirical_quantiles(
    fig,
    ax,
    grid,
    green_curve,  # ~~~ tensor with the same shape as `grid`
    x_train,
    y_train,  # ~~~ tensor with the same shape as `x_train`
    predictions,  # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
    extra_std,
    how_many_individual_predictions,
    title,
    **kwargs,
):
    return univar_figure(
        fig,
        ax,
        grid,
        green_curve,
        x_train,
        y_train,
        model="None! All we need is the matrix of predictions",
        title=title,
        blue_curve=lambda model, grid, ax: empirical_quantile(
            predictions,
            grid,
            ax,
            extra_std,
            how_many_individual_predictions=how_many_individual_predictions,
        ),
        **kwargs,
    )


#
# ~~~ Load a trained model, based on the dataframe of results you get from hyperparameter search, and then plot it
def plot_trained_model_from_dataframe(
    dataframe, i, n_samples=50, show=True, visualize_using_quantiles=False, extra_std=True, val=False, title=None, **other_kwargs
):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    grid = data.x_test.cpu()
    green_curve = data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu().squeeze()
    y_train_cpu = data.y_train.cpu().squeeze()
    x_val_cpu = data.x_val.cpu().squeeze()
    y_val_cpu = data.y_val.cpu().squeeze()
    model = load_trained_model_from_dataframe(dataframe, i)
    if show:
        visualize_using_quantiles = dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES if (visualize_using_quantiles is None) else visualize_using_quantiles
        plot_predictions = (
            plot_bnn_empirical_quantiles
            if visualize_using_quantiles
            else plot_bnn_mean_and_std
        )
        with torch.no_grad():
            try:
                predictions = model(grid, n=n_samples).squeeze()
            except TypeError:
                predictions = model(grid).squeeze()
            fig, ax = plt.subplots(figsize=(12, 6))
            fig, ax = plot_predictions(
                fig=fig,
                ax=ax,
                grid=grid,
                green_curve=green_curve,
                x_train=x_val_cpu if val else x_train_cpu,
                y_train=y_val_cpu if val else y_train_cpu,
                predictions=predictions,
                how_many_individual_predictions=0,
                # dataframe.iloc[
                #     i
                # ].HOW_MANY_INDIVIDUAL_PREDICTIONS,
                extra_std=extra_std,
                title=title or f"Trained Model i={i}/{len(dataframe)}",
                **other_kwargs,
            )
            plt.show()
    return model, x_train_cpu, y_train_cpu, grid, green_curve


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/my_base_utils.py
### ~~~


#
# ~~~ Compute [min-c,max+c] where c>0 is a buffer
def buffer(vector, multiplier=0.05):
    a = min(vector)
    b = max(vector)
    extra = (b - a) * multiplier
    return [a - extra, b + extra]


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/my_plt_utils.py
### ~~~


try:
    get_ipython()
    from IPython.display import clear_output, display
    from IPython.display import Image as colab_image

    this_is_running_in_a_notebook = True
except NameError:
    this_is_running_in_a_notebook = False


#
# ~~~ A simple function which closes any and all open matplotlib figures
def close_all_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()


#
# ~~~ A tool for making gif's
class GifMaker:
    #
    # ~~~ Instantiate what is essentially just a list of images
    def __init__(self, path_or_name="my_gif", ram_only=True, live_frame_duration=0.01):
        #
        # ~~~ Baseic attributes
        self.frames = []  # ~~~ the list of images
        self.ram_only = ram_only  # ~~~ boolean, whether to store images only in RAM (instead of using the disk)
        path_or_name = (
            os.path.join(os.getcwd(), path_or_name)
            if os.path.dirname(path_or_name) == ""
            else path_or_name
        )
        self.too_many_figures = (
            []
        )  # ~~~ a safety feature; here we store flags that may get triggered
        self.live_frame_duration = (
            live_frame_duration  # ~~~ how long to show each frame for the user, live
        )
        #
        # ~~~ I was getting unexpected results from live_frame_duration=0, but a simple `if not live_frame_duration>0` clause fails to account for `live_frame_duration=None`, so I used try/except
        try:
            assert self.live_frame_duration > 0
        except:
            self.live_frame_duration = None
        #
        # ~~~ Save a master path to be used by default for this gif
        self.master_path = process_for_saving(
            os.path.splitext(path_or_name)[0]
        )  # ~~~ strip any file extension if present, and modify the file name if necessary to avoid save conflicts
        #
        # ~~~ If we don't want to store images only in RAM, then create a folder in which to store the pictures temporarily
        if not ram_only:
            self.temp_dir = process_for_saving(self.master_path + " temp storage")
            os.mkdir(self.temp_dir)

    #
    # ~~~ Safety feature of sorts which checks how many figures are currently open *and* complains and makes a note if that number is not exactly 1
    def check_how_many_figs_are_open(self):
        number_of_figs_currently_open = len(plt.get_fignums())  # ~~~ check how many
        if not number_of_figs_currently_open == 1:
            my_warn(
                f"capture() method expects exactly 1 figure to be open, however {number_of_figs_currently_open} figures were found."
            )  # ~~~ complain
            self.too_many_figures.append(
                number_of_figs_currently_open
            )  # ~~~ make a note

    #
    # ~~~ Method that, when called, saves a picture of whatever would be returned by plt.show() at that time
    def capture(self, clear_frame_upon_capture=True, **kwargs):
        #
        # ~~~ Save the figure either in RAM (if `ram_only==True`) or at a path called `filename`
        temp = BytesIO() if self.ram_only else None
        filename = (
            None
            if self.ram_only
            else process_for_saving(os.path.join(self.temp_dir, "frame (1).png"))
        )
        plt.savefig(temp if self.ram_only else filename, **kwargs)
        #
        # ~~~ Add to our list of pictures (called `frames`), either the picture that's in RAM (if `ram_only==True`) or the path from which the picture can be loaded
        self.frames.append(temp.getvalue() if self.ram_only else filename)
        #
        # ~~~ Show the current frame to the user, if self.live_frame_duration is not None
        if self.live_frame_duration is not None:
            if this_is_running_in_a_notebook:
                display(colab_image(self.frames[-1]))
                clear_output(wait=True)
            else:
                plt.draw()
                plt.pause(self.live_frame_duration)
        #
        # ~~~ A safety feature of sorts: check whether or not the number of open figures seems to be growing
        self.check_how_many_figs_are_open()  # ~~~ if more than 1 fig is open, then the number of open figures will be appended to the attribute `self.too_many_figures`
        number_of_figures_seems_to_be_growing = (
            len(self.too_many_figures) >= 3
            and np.diff(self.too_many_figures).mean() > 0.2
        )  # ~~~ `0.2` would imply that a new figure is created on average once each 5 times `capture()` is called
        if number_of_figures_seems_to_be_growing and (
            self.live_frame_duration is not None
        ):
            my_warn(
                f"At least twice, in between calls to `capture()`, the number of open figures has increased. This almost certainly means that new figures are being created in between calls to the `capture()` method, which interferes with live rendering of the gif. Therefore, live rendering of the gif will be deactivated. New frames will continue to be added to the gif, but they won't be shown live going forward."
            )
            self.live_frame_duration = (
                None  # ~~~ setting this to `None` disables interactive plotting
            )
        #
        # ~~~ If clear_frame_upon_capture==True, attempt to delete (in the appropriate manner) the picture that we just saved
        if (
            clear_frame_upon_capture
        ):  # ~~~ setting `clear_frame_upon_capture=False` disables all of these ad-hoc shenanigans
            if self.live_frame_duration is None:
                plt.clf()  # ~~~ if we're not concerned about interactive plotting, then just clear the frame plain and simple
            else:
                current_fig = plt.figure(plt.get_fignums()[-1])
                axs_of_current_fig = current_fig.get_axes()
                for ax in axs_of_current_fig:
                    ax.clear()  # ~~~ closing the frame could actually interfere with interactive ploting; therefore, instead, try to just clear the *axes* of the frame
            if number_of_figures_seems_to_be_growing:
                close_all_figures()  # ~~~ on the other hand, if it seems like the number of figures is accumulating, try to curtail this by closing all figures

    #
    # ~~~ Delete the individually saved PNG files and their temp directory
    def cleanup(self):
        if not self.ram_only:
            for file in self.frames:
                if os.path.exists(file):
                    os.remove(file)
            os.rmdir(self.temp_dir)
        del self.frames
        self.frames = []

    #
    # ~~~ Method that "concatenates" the list of picture `frames` into a .gif
    def develop(
        self,
        destination=None,
        total_duration=None,
        fps=None,
        cleanup=True,
        verbose=True,
        loop=0,
        **kwargs,
    ):
        #
        # ~~~ Process individual frames
        if self.ram_only:
            #
            # ~~~ Convert raw bits back to the image they were encoded from
            images = [Image.open(BytesIO(temp)) for temp in self.frames]
        else:
            #
            # ~~~ Load the image from the path that it was saved to
            images = [Image.open(file) for file in self.frames]
        #
        # ~~~ Process destination path
        destination = (
            self.master_path if destination is None else destination
        )  # ~~~ default to the path used for temp storage (which, itself, defaults to os.getcwd()+"my_gif")
        destination = (
            os.path.join(os.getcwd(), destination)
            if os.path.dirname(destination) == ""
            else destination
        )  # ~~~ if the destination path is just a filename, consider it as a file within the os.getcwd()
        destination = process_for_saving(
            destination.replace(".gif", "") + ".gif"
        )  # ~~~ add `.gif` if not already preasent, turn "file_name.gif" into "file_name (1).gif", etc.
        #
        # ~~~ Infer a frame rate from the desired duration, if the latter is supplied
        fps_was_None = fps is None
        if fps_was_None:
            fps = 30
        if total_duration is None:
            total_duration = len(self.frames) / fps
        else:
            fps = len(self.frames) / total_duration
            if fps_was_None:
                my_warn(
                    "Both `total_duration` and `fps` were supplied (note `fps` has a default value of 30). `fps` will be ignored."
                )
        total_duration = float(total_duration)
        #
        # ~~~ Save the thing
        if verbose:
            print(f"Saving gif of length {total_duration:.3} sec. at {destination}")
        images[0].save(
            destination,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=loop,
            **kwargs,
        )
        self.desination = destination  # ~~~ store the save directory as an attribute for later reference
        #
        # ~~~ Clean up the workspace if desired
        if cleanup:
            plt.close()
            self.cleanup()


#
# ~~~ Renders the image of a scatter plot overlaid with the graphs of one of more functions (I say "curve" because I'm remembering the R function `curve`)
def points_with_curves(
    x,
    y,
    curves,  # ~~~ (m1,m2,m3,...,ground_truth)
    points_label=None,
    curve_colors=None,
    marker_size=None,
    marker_color=None,
    point_mark=None,
    curve_thicknesses=None,
    curve_labels=None,
    curve_marks=None,
    curve_alphas=None,
    grid=None,
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    crop_ylim=True,
    tight_layout=True,
    show=True,
    legend=True,
    fig="new",
    ax="new",
    figsize=(6, 6) if this_is_running_in_a_notebook else None,
    model_fit=True,  # default usage: the plot to be rendered is meant to visualize a model's fit
):
    #
    # ~~~ If x and y are pytorch tensors on gpu, then move them to cpu
    try:
        x = x.cpu()
        y = y.cpu()
    except AttributeError:
        pass
    #
    # ~~~ Automatically set default values for several arguments
    n_curves = len(curves)
    if points_label is None:
        points_label = "Training Data"
    if marker_size is None:
        marker_size = 4 if len(x) < 400 else max(247 / 60 - 7 / 24000 * len(x), 1.2)
    if marker_color is None:
        marker_color = "green"
    if point_mark is None:
        point_mark = "o"
    if curve_colors is None:
        curve_colors = ("hotpink", "orange", "midnightblue", "red", "blue", "green")
    if curve_alphas is None:
        curve_alphas = [
            1,
        ] * n_curves
    if xlim is None:
        xlim = buffer(x, multiplier=0.05)
    if grid is None:
        grid = torch.linspace(min(xlim), max(xlim), 1001)
    if ylim is None and crop_ylim:
        ylim = buffer(y, multiplier=0.2)
    curve_thicknesses = (
        [min(0.5, 3 / n_curves)] * n_curves
        if curve_thicknesses is None
        else curve_thicknesses[:n_curves]
    )
    #
    # ~~~ Facilitate my most common use case: the plot to be rendered is meant to visualize a model's fit
    if (curve_labels is None) and (curve_marks is None) and model_fit:
        curve_labels = (
            ["Fitted Model", "Ground Truth"]
            if n_curves == 2
            else [f"Fitted Model {i+1}" for i in range(n_curves - 1)] + ["Ground Truth"]
        )
        curve_marks = ["-"] * (n_curves - 1) + ["--"]  # ~~~ make the last curve dashed
        curve_colors = curve_colors[
            -n_curves:
        ]  # ~~~ make the last curve green, which matches the default color of the dots
    elif model_fit:
        arg_vals, arg_names = [], []
        if curve_labels is not None:
            arg_names.append("curve_labels")
            arg_vals.append(str(curve_labels))
        if curve_marks is not None:
            arg_names.append("curve_marks")
            arg_vals.append(str(curve_marks))
        if not n_curves == 2:
            arg_names.append("n_curves")
            arg_vals.append(str(n_curves))
        warning_msg = (
            "the deafult `model_fit=True` is overriden by user_specified input(s):"
        )
        for i, (name, val) in enumerate(zip(arg_names, arg_vals)):
            warning_msg += " " + name + "=" + val
            if i < len(arg_vals) - 1:
                warning_msg += " |"
        my_warn(warning_msg)
    #
    # ~~~ Other defaults and assertions
    curve_labels = (
        [f"Curve {i}" for i in range(n_curves)]
        if curve_labels is None
        else curve_labels[:n_curves]
    )
    curve_marks = ["-"] * n_curves if curve_marks is None else curve_marks[:n_curves]
    assert curve_thicknesses is not None
    assert curve_labels is not None
    assert curve_marks is not None
    assert grid is not None
    assert len(x) == len(y)
    assert len(curves) <= len(curve_labels)
    assert len(curves) <= len(curve_colors)
    assert (fig == "new") == (ax == "new")  # either both new, or neither new
    #
    # ~~~ Do the thing
    fig, ax = (
        plt.subplots(figsize=figsize) if (fig == "new" and ax == "new") else (fig, ax)
    )  # supplied by user in the latter case
    ax.plot(
        x,
        y,
        point_mark,
        markersize=marker_size,
        color=marker_color,
        label=(points_label if legend else ""),
    )
    with torch.no_grad():
        for i in range(n_curves):
            try:  # ~~~ try to just call curves[i] on grid
                curve_on_grid = curves[i](grid)
            except:
                try:  # ~~~ except, if that doesn't work, then assume we're in pytorch,
                    assumed_device = (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )  # ~~~ assume that the curve is on "the best device available"
                    curve_on_grid = curves[i](grid.to(assumed_device)).cpu()
                except:
                    raise ValueError(
                        "Unable to evaluate `curve(grid)`; please specify `grid` manually in `points_with_curves` and/or verify the definitions of the arguments `grid` and `curves` in `points_with_curves`"
                    )
            #
            # ~~~ Transfer grid and curve_on_grid to cpu, if they are pytorch tensors on gpu
            grid = grid.cpu() if hasattr(grid, "cpu") else grid
            curve_on_grid = (
                curve_on_grid.cpu() if hasattr(curve_on_grid, "cpu") else curve_on_grid
            )
            ax.plot(
                grid,
                curve_on_grid,
                curve_marks[i],
                curve_thicknesses[i],
                color=curve_colors[i],
                label=(curve_labels[i] if legend else ""),
                alpha=curve_alphas[i],
            )
    #
    # ~~~ Further aesthetic configurations
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    #
    # ~~~ The following lines replace `plt.legend()` to avoid duplicate labels; source https://stackoverflow.com/a/13589144
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = list(set(labels))  # Get unique labels
        by_label = (
            {}
        )  # Create a dictionary to store handles and line styles for each unique label
        for label in unique_labels:
            indices = [
                i for i, x in enumerate(labels) if x == label
            ]  # Find indices for each label
            handle = handles[
                indices[0]
            ]  # Get the handle for the first occurrence of the label
            line_style = handle.get_linestyle()  # Get the line style
            by_label[label] = (handle, line_style)  # Store handle and line style
        legend_handles = [by_label[label][0] for label in by_label]
        legend_labels = [
            f"{label}" for label in by_label
        ]  # Include line style in label
        plt.legend(legend_handles, legend_labels)
    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax
