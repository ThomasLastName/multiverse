# Setup

## Setup steps using anaconda

(without anaconda, just make sure you have the prerequisite packages installed)

0. Open the terminal and say `conda env list` to confirm that the code is not present already.

1. (_create an env with standard / easy-to-install packages_) `conda create --name bnns python=3.10 tqdm matplotlib numpy pandas plotly scipy pip` (if desired, you can swap `bnns` for your preferred name).

2. (_activate the env for further installs_) `conda activate bnns`.

3. (_install pytorch_) This may depend on whether you want cuda, and on your conda channels. The simplest approach is: first try `conda install pytorch`. If that doesn't work (probably because channels) then try instead `pip install torch`.

4. (_install this code_) Navigate to wherever you want (e.g., the Documents folder), and clone this repo there. Then (mimicing [the SEPIA installation guidelines](https://sepia-lanl.readthedocs.io/en/latest/#installation)), "from the command line, while in the [the root depository of this repository], use the following command to install [bnns]:" `pip install -e .` "The -e flag signals developer mode, meaning that if you update the code from Github, your installation will automatically take those changes into account without requiring re-installation."

5. (_verify installation_) Try running one of the python files, e.g., `python scripts/SSGE_multivar_demo.py`, which should create a .gif of some histograms.


## Dependencies

Well, you need pytorch and matplotlib and such.
Perhaps non-trivially you need tqdm.
**Most notably,** you need my helper utils https://github.com/ThomasLastName/quality_of_life which you just need clone to anywhere on the path for your python environment (I got the impression from Natalie that y'all are allowed clone repos off the internet to your lanl devices? You need this repo)

I believe, the complete list of required dependencies, excluding the standard library (e.g., `typing`) is:
- [ ] pytorch
- [ ] matplotlib
- [ ] tqdm
- [ ] numpy
- [ ] scipy
- [ ] plotly
- [ ] pyreadr
- [ ] fiona
- [ ] https://github.com/ThomasLastName/quality-of-life (this repo has its own dependencies, but I believe it is sufficient to run this repo with only the above packages installed; I believe "the required parts" of this repo depend only on the same 5 packages as above and the standard python library).

If desired, the dependencies on `plotly` and `quality_of_life` could be removed.

Once you've activated the desired virtual env, you can install the dependencies by running the following command inside the `functional_bnns/` directory:

```
pip install -r requirements_fbnns.txt
```

# Usage

In order to run a test, the procedure is as follows. In order to specify hyperparameters, put a `.json` file containing hyperparameter values for the experiment that you want to run in the `experiments` folder.
Different algorithms require different hyperparmeters, and these differences are reflected in the scripts that load the `.json` files.
At the time of writing, there are 4 python scripts in the `experiments` folder: `train_bnn.py`, `train_nn.py`, `train_gpr.py`, and `train_ensemble.py`. To train a model with the hyperparamters specified by the `.json` file, say, `my_hyperpars.json`, navigate to the `experiment` folder and run `python train_<algorithm>.py --json my_hyperparameters`.
To see which hyperparameters are expected by the algorithm (which are the fields that you need to include in your .json file), check either the demo .json file included with the repo, or check the body of the python script, where a dictionary called `hyperparameter_template` should be defined.

## The SLOSH Dataset

The SLOSH dataset can only by used if you have the file `slosh_dat_nj.rda` located in the `experiments` folder (**not included with the repo!**).

The Y data of the slosh data set has m=4000 rows and n_y>49000 columns. Instead of training on the full dataset, we follow the PCA decomposition of [https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2796](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2796), which we now review.
Assume that the matrix $`Y`$ is a "data matrix," in the sense that each datum is a *row* of the matrix (as opposed to a column).
Then, in the SVD

```math
    Y = USV^\intercal = \sum_{k \leq r} s_k u^{(k)} \big[v^{(k)}\big]^\intercal,
````

the $`(\ell,j)`$-th entry of $`Y`$ is

```math
    Y_{\ell,j} = \sum_{k \leq r} s_k u_\ell^{(k)} v_j^{(k)}.
````

From this, we can see each original datum (each row of $`Y`$) is a linear combination of the vectors $`s_1v^{(1)}, \ldots, s_rv^{(r)}`$.
The dependence on $\ell$ occurs only through the coefficients $`u_\ell^{(1)}, \ldots, u_\ell^{(r)}`$ of this linear combination.
The interpretation is as follows:
 - The *right* singular vectors $`v^{(1)}, \ldots, v^{(r)}`$ are the "principal heatmaps." Every heay map in our dataset (i.e., every row of the data matrix $`Y`$) is a linear combination of them.
 - The coefficients $`s_1,\ldots,s_r`$ are expected to be have $`s_k \approx 0`$ for $`k`$ large. They are not included in the "principal heatmaps." Rather, they merely *down-weight* the "principal heatmaps." 
 - The m-by-r matrix $`U`$ of *left* singular vectors is what needs to be predicted. They are what varies from sample to sample, for they are all that depends on the row index $`\ell`$ of the data matrix.
 - When a vector of coefficients $`(a_1,\ldots,a_r)`$ is produced by some predictive model, the final prediction is $`a_1 s_1 v^{(1)} + ... + a_r s_r v^{(r)}`$, which has the same shape (and meaning!) as one of the rows of $Y$. Thus, given a batch `A` of such vectors, i.e., a matrix with $`r`$ columns, the final batch of predictions is given by $`A S V^\intercal`$.

In other words, the originally given data matrix `Y` is pre-processed with an SVD `Y = U @ S @ V.T` where `S` is diagonal. Then `S` and `V` are stored for the prediction phase, while the processed matrix `U` is treated as the data matrix for the purposes of training.
After training, a batch prediction `P` with as many columns as `U` (but fewer rows: only as many as the batch size) can be re-converted into the same format as `Y` via `final_prediction = P @ S @ V.T`, each *row* of which should look like "the same kind of data" as each row of `Y`.
If this procedure is applied directly to the matrix `Y = y_train`, then the "kind of data" in question would be a heatmap, like what one sees visualized in aforementioned paper.
However, note that the matrix `Y` could, itslef, have already been a processed version of the data (e.g., subtracint the mean), in which case the final predicted heatmap would also require further processing to reflect/undo however `Y` was obtrained from "the real data."
That is to say, one must be cognizant of whether or not PCA is *the only* pre-processing that's done to the data.

## Creating your own Dataset

All the .json files are supposed to have a field called "data" whose value is a text string. Suppose the "data" field has a value of "my_brand_new_dataset".
In that case, the python scripts which run experiments all attempt to `import my_brand_new_dataset from bnns.data` meaning that you need to create a file called `my_brand_new_dataset.py` located in the folder `data` if you want this to work.
Within the file `my_brand_new_dataset.py`, you must define 3 pytorch datasets: `D_train`, `D_val`, and `D_test`, as well as two pytorch tensors `interpolary_grid` and `extrapolary_grid`. The python scripts that run experiments will attempt to access these variables from that file in that location.
Additionally, for examples with a one-dimensional input, if you want the scripts to plot your models, then you must define a pytorch vector `grid` with `grid.ndim==1` which is used to create the plots.

## Creating your own Models

All the .json files are supposed to have a field called "model" whose value is a text string. Suppose the "model" field has a value of "my_brand_new_architecture".
In that case, the python scripts which run experiments all attempt to `import my_brand_new_architecture from bnns.data` meaning that you need to create a file called `my_brand_new_architecture.py` located in the folder `models` if you want this to work.
Additionally, within that file `my_brand_new_architecture.py`, you must define a pytorch model: either called `BNN` or called `NN` depending on the experiment that is being run


# Contributors

 - The code for SSGE was adapted from the repo https://github.com/AntixK/Spectral-Stein-Gradient



# TODO

See the Issues tab.
