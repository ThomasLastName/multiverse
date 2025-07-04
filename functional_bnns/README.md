# Summary

**IMPORTANT: At this time, the package is still in development, and is not yet ready for use by a general audience.**

## Main Features

This package fulfills a need for reliable, modular, general, and efficient open source implementations of variational Bayesian neural networks (BNNs).
Specifically, this package offers the following 3 general features.

### 1. Implementations of Training Methods Other than BBB (TODO: LINK TO TUTORIAL)

The training methods implemented in this package are:
 - BBB ([Blundell et al. 2015](https://arxiv.org/pdf/1505.05424))
 - SVGD                     (citation needed)
 - The original fBNN method (citation needed)
 - Gaussian approximation   (citation needed)
 - (pending) fSVGD          (citation needed)

as well as, for the sake of comparison,
 - Deterministic neural networks
 - Neural networks with dropout
 - Conventional neural network ensembles
 - Conventional Gaussian process regression

### 2. A Flexible, Canonical Framework for Custom User-Defined BNNs (TODO: LINK TO TUTORIAL)

The base class `bnns.BayesianModule` is intended to provide a similar level of flexibility as its super-class `nn.Module`.
Just as PyTorch provides the ready-to-use sub-class `nn.Sequential` of `nn.Module`, for convenience, this package also provides several ready-to-use sub-classes of `bnns.BayesianModule` (TODO refer to them by name):
 - Mutually independent normally distributed weights with
   * an independent normal prior distribution over those weights
   * a Gaussian process prior over network outputs
   * (pending) other location-scale priors over those weights with full support
   * (pending) the Gaussian mixture prior proposed in [Blundell et al. 2015](https://arxiv.org/pdf/1505.05424)
 - (pending) Mutually independent uniformly distributed weights
   * (pending) an independent uniform prior distribution over those weights
   * (pending) other location-scale priors over those weights with full support

### 3. A Minimalist Infrastructure for Hyper-Parameter Tuning and Model Benchmarking (TODO: LINK TO TUTORIAL)

For our own experiments, we have devloped a basic API for training models.
The folder `multiverse/functional_bnns/bnns/experiments/` containts four scripts for training various models:
 - `train_nn.py` fits a neural network, with or without dropout
 - `train_bnn.py` fits a Bayesian neural network, supporting a variety of different training methods and priors
 - `train_ensemble.py` fits a neural network ensemble, either by conventional means, SVGD, or fSVGD
 - `train_gpr.py` fits a Gaussian process

To train a model with the hyperparamters specified in a file `my_hyperpars.json`, navigate to the `experiments` folder and run `python train_<algorithm>.py --json my_hyperpars`.
The required fields vary slightly between these four scripts.
For more, see [__Training, Tuning, and Testing Infrastructure__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#training-tuning-and-testing-infrastructure) below.

For hyperparameter tuning and model benchmarking, this package's approach is to populate a folder with `.json` files (one per. hyper-parameter configuration you wish to test), and run the training script for each `.json` file.
A script called `tuning_loop.py` automates this process.
This framework allows one to easily replicate our own experiemnts.
For more, see [__Paper__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#paper) below.

## Purpose

This package was developed in an effort to answer several research questions of increasing scope: (i) are "functional priors" better than "weight priors," (ii) what's the best way to train a BNN, and (iii) do BNN's even work all that well, frankly?

For context, there are many possible ways to train a BNN, of which BBB is by far the most popular.
In Bayesian modeling of any kind, the prior distribution is always extremely important.
However, BBB supports only a limited variety of prior distributions (viz. ones specified on the weights of a neural network: "weight priors").
Several of our scientists and many authors have hypothesized that "functional priors" may yield better predictive posterior models than weight priors.
To test that hypothesis, at bare minimum, we thus required an implementation of some training method which supports functional priors.

Moreover, we required an implementation *at a sufficient level of quality* to also support extensive and varied empirical experiments.
Finally, in order to "control for" the role of the training method and "isolate" the role of the prior distribution, it would be uncscientific to test only 1 or 2 training methods.
Rather, we felt that the most rigorous approach would be to implement several different training methods, in order to be confident that our conclusions are really the result of the prior distribution, and not merely an artifact of training.
Unfortunately, production-level open source implementations existed only of BBB, and not (to our awareness) of the other training methods we wished to test.
In fact, the prevalence of BBB above all other training methods may be merely a spurious result of the availability of software.
Consequently, this package, also, serves to test whether or not BBB deserves its tentative status as the default method to train a BNN.

So that is _why_ this package provides implementations of training methods other than BBB.
At the same time, modularizing the code and ensuring its flexibility became a necessity once we ramped up the variety of different architectures and priors we sought to test; that's _why_ this package provides a flexible, canonical framework for custom user-defined BNNs.
Finally, since the original purpose of this package was to test rigorously which type of prior distribution gives the most desirable outcomes, of course a testing pipeline was necessary, as was a bunch of hyper-parameter tuning; that's _why_ this package provides a minimalist infrastructure for hyper-parameter tuning and model benchmarking.

---

# Setup

## Requirements

**IMPORTANT: At this time, `git` is a prerequisite for installation.**

Due to `GPyTorch` being added as a dependency, Python 3.8 or later is now required.


## Minimal setup instructions using git

1. (_have python already set up_) Basically, just have python on your machine in an environment that allows you to install packages. Optionally, this may include setting up a virtual environment for this repository and its dependencies, which most programmers would opt to do. The below **Setup steps using anaconda and git** go into more detail on this matter, walking you through the process of setting up such an environment using `conda`.

2. (_have pytorch already installed_) Technically, this step can be skipped. The final `pip install` step will automatically make sure that _some_ version of `PyTorch >= 2.0` is installed as a dependency. However, if a partircular `PyTorch` distribution is necessary for CUDA-compatibility. Now's the time to install that.

3. (_clone this repo_) Navigate to wherever you want this code to be stored (e.g., the Desktop, or the Documents folder), and clone this reppository there using `git clone https://github.com/ThomasLastName/multiverse.git` in the command line.

4. (_install the `multiverse/functional_bnns` directory as a package_) Navigate into the folder containing `setup.py` and, from that directory, run `pip install -e .` in the command line.


## Setup steps using anaconda, pip, and git

0. Open the command line and say `conda env list` to confirm that the code is not present already.

1. (_create an env with standard / easy-to-install packages_) `conda create --name bnns python=3.10 matplotlib tqdm numpy scipy pandas seaborn pip` (if desired, you can swap `bnns` for your preferred name).

2. (_activate the env for further installs_) `conda activate bnns`.

3. (_install pytorch_) This dependency is intentionally left to the user to be installed manually, because the appropriate version of `torch` may depend on your hardwarde, particularly on CUDA-compatibility. Additionally, it may depend on your conda channels. The simplest installation (which is not CUDA-compatible) is to try the command `conda install pytorch`. If that doesn't work (probably because of channels) then commanding `pip install torch` while the environment is active shuold also work, although using `conda` is preferable because it reduces the likelihood of conflicts.

4. (_clone this repo_) Navigate to wherever you want this code to be stored (e.g., the Desktop, or the Documents folder), and clone this reppository there using `git clone https://github.com/ThomasLastName/multiverse.git` in the command line.

5. (_install the `multiverse/functional_bnns` directory as a package_) Navigate into the folder containing `setup.py` and, from that directory, run `pip install -e .` in the command line. As in [the SEPIA installation guidelines](https://sepia-lanl.readthedocs.io/en/latest/#installation), "the -e flag signals developer mode, meaning that if you update the code from Github, your installation will automatically take those changes into account without requiring re-installation."

6. (_verify installation_) Try running one of the python files, e.g., `python scripts\SSGE_univar_demo.py`, which should create an plot with several curves.

---

# Organization of the Package

## Folder Structure

The `functional_bnns` folder contains only two folders: `scripts` which consists of some rudiementary tests implemented during development and can be ignored by users, and `bnns` which contains the actual guts of the package.
Within the important one of these two folders, the package is organized as follows. More details can be found elsewhere in this README

 - `data/` folder containing some synthetic data and code  which downloads non-synthetic data used in our experiments
 - `experiments/` folder containing infrastructure for training, tuning, and testing models ([__Training, Tuning, and Testing Infrastructure__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#training-tuning-and-testing-infrastructure))
   * `paper/` contains files for running/replicating our expiments ([__Paper__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#paper))
   * `train_nn.py` trains a neural net
   * `train_bnn.py` trains a Bayesian neural net
   * `train_ensemble.py` trains a neural network ensemble
   * `train_gpr.py` fits a Gaussian process
   * `demo_nn.json` ready-made example of an appropriately structured `.json` file
   * `demo_bnn.json` ready-made example of an appropriately structured `.json` file
   * `demo_ensemble.json` ready-made example of an appropriately structured `.json` file
   * `demo_gpr.json` ready-made example of an appropriately structured `.json` file
 - `models/` folder containing code that defines (untrained versions of) the models used in our experiments
 - `__init.__py` boiler plate
 - `Ensemble.py` defines classes for efficient (parallelized) neural network ensembles, including support for SVGD and fSVGD
 - `GPPriorBNNs.py` sub-classes the base classes of `NoPriorBNNs.py` with the logic for GP priors ([__Class Structure__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#class-structure))
 - `GPR.py` defines classes for Gaussian processes
 - `metrics.py` defines functions for model benchmarking
 - `NoPriorBNNs.py` defines base classes for BNNs, which expect sub-classes to add extra logic needed for prior distributions ([__Class Structure__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#class-structure))
 - `SequentialGaussianBNN.py`
 - `SSGE.py` defines class for the spectral stein gradient estimator used in conventional functional BNN training
 - `utils.py` defines various mundane helper functions
 - `WeightPriorBNNs.py` sub-classes the base classes of `NoPriorBNNs.py` with the logic for various differnet weight priors ([__Class Structure__](https://github.com/ThomasLastName/multiverse/tree/main/functional_bnns#class-structure))

## Class Structure

TODO


---

# Training, Tuning, and Testing Infrastructure

In order to run a test, the procedure is as follows. In order to specify hyperparameters, put a `.json` file containing hyperparameter values for the experiment that you want to run in the `experiments` folder.
Different algorithms require different hyperparmeters, and these differences are reflected in the scripts that load the `.json` files.
At the time of writing, there are 4 python scripts in the `experiments` folder: `train_bnn.py`, `train_nn.py`, `train_gpr.py`, and `train_ensemble.py`. To train a model with the hyperparamters specified by the `.json` file, say, `my_hyperpars.json`, navigate to the `experiment` folder and run `python train_<algorithm>.py --json my_hyperparameters`.
To see which hyperparameters are expected by the algorithm (which are the fields that you need to include in your .json file), check either the demo .json file included with the repo, or check the body of the python script, where a dictionary called `hyperparameter_template` should be defined.


Required fields include, `"MODEL"` (the directory of a `.py` file from which to load the pytorch `Module`), `"EPOCHS"` (how many epochs to train for), etc.
To see which fields are required, check either the body of the training script, where a dictionary called `hyperparameter_template` is defined, or the demo `.json` files that are also found in the `expierments` folder:
 - `demo_nn.json` (run with `python train_nn.py --json demo_nn`)
 - `demo_bnn.json` (run with `python train_bnn.py --json demo_bnn`)
 - `demo_ensemble.json` (run with `python train_ensemble.py --json demo_ensemble`)
 - `demo_gpr.json` (run with `python train_gpr.py --json demo_gpr`)


## Creating your own Dataset

All the .json files are supposed to have a field called "data" whose value is a text string. Suppose the "data" field has a value of "my_brand_new_dataset".
In that case, the python scripts which run experiments all attempt to `import my_brand_new_dataset from bnns.data` meaning that you need to create a file called `my_brand_new_dataset.py` located in the folder `data` if you want this to work.
Within the file `my_brand_new_dataset.py`, you must define 3 pytorch datasets: `D_train`, `D_val`, and `D_test`, as well as two pytorch tensors `interpolary_grid` and `extrapolary_grid`. The python scripts that run experiments will attempt to access these variables from that file in that location.
Additionally, for examples with a one-dimensional input, if you want the scripts to plot your models, then you must define a pytorch vector `grid` with `grid.ndim==1` which is used to create the plots.

## Creating your own Models

All the .json files are supposed to have a field called "model" whose value is a text string. Suppose the "model" field has a value of "my_brand_new_architecture".
In that case, the python scripts which run experiments all attempt to `import my_brand_new_architecture from bnns.data` meaning that you need to create a file called `my_brand_new_architecture.py` located in the folder `models` if you want this to work.
Additionally, within that file `my_brand_new_architecture.py`, you must define a pytorch model: either called `BNN` or called `NN` depending on the experiment that is being run



# Paper

## Replicating Our Experiments

Our experiments were run using the Training, Tuning, and Testing API described above.
The sub-directory `multiverse/functional_bnns/bnns/experiments/paper` of the `experiments` folder contains what little code is needed to run experiments using that API.
Specifically, there is a sub-directory for each dataset that we have tested:

 - `univar/` folder for all the tests we run on a simple, synthteic univariate regression dataset
 - `SLOSH/` (pending) folder for all the tests we run on the SLOSH dataset

For each dataset, we run the following tests in the order listed:

 1. (`det_nn/`) We begin with 16 different neural network architectures, and fit these to the dataset by vanilla, deterministic training. By tuning only learning rate, we narrow down which architectures to test further by taking the best 8.
 2. (`dropout/`) We test the remaining 8 architectures with various dropout levels. By tuning the learning rate and level of dropout, we narrow down which architectures to test further by taking the best 4.
 3. (`weight_training_vs_functional_training/`) We test BNN training methods on the remaining 4 architectures in order to verify the hypothesis that "only the prior matters, and the exact training hyper-parameters hardly matter." Testing this hypothesis greatly simplifies the matter of tuning for the effects of differnt prior distributions, as we no longer need to worry about "controlling for" mis-specified training settings and can focus on just tuning the prior going forward. We use the results of this experiment to select training hyper-parameters for futher tests.
 4. .......

Each of these containins the following folders:

 - `det_ensemble/` (pending) test the best 1/4 of architectures with a neural network ensemble
 - `det_nn/` test a handful of different architectures on the dataset, and slecting the best half
 - `dropout/` test the best half of architectures with dropout, and further selecting the best half
 - `stein_ensemble` (pending) test the best 1/4 of architectures with SVGD
 - `weight_training_vs_functional_training` tests the hypothesis that functional and non-functional training give identical resutls

Finally, each of those contains the following:

 - `__init__.py` defines the main settings to be tested in the experiment at hand (imported in `__main__.py`) 
 - `__main__.py` creates and populates a folder `hyper_parameter_search/` full of `.json` files for all cases to be tested
 - `process_results.py` processes the raw data with which the folder `hyper_parameter_search/` will be filled

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


# Contributors

 - The code for SSGE was adapted from the repo https://github.com/AntixK/Spectral-Stein-Gradient


# Contribution Guidelines

TODO

