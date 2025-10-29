import os
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bnns.utils import infer_width_and_depth, plot_trained_model_from_dataframe, my_warn
from bnns.experiments.paper.univar.ensemble_for_baseline import (
    folder_name,
    DATA,
    ARCHITECTURE,
)


### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, processed in a format that pandas likes (remove any lists), and combined into a pandas DataFrame
### ~~~

folder_dir = os.path.split(folder_name)[0]
try:
    results = pd.read_csv(os.path.join(folder_dir, "results.csv"))
except FileNotFoundError:
    from bnns.utils import load_filtered_json_files
    results = load_filtered_json_files(folder_name)
    results.to_csv(os.path.join(folder_dir, "results.csv"))
except:
    raise


### ~~~
## ~~~ Process the slightly more
### ~~~

results = infer_width_and_depth(results)

#
# ~~~ Verify that DATA==results.DATA.unique(), ARCHITECTURE==results.ARCHITECTURE.unique(), and LR==results.LR.unique()
if len(DATA) == 2 == len(results.DATA.unique()) and len(ARCHITECTURE) == 12 == len(
    results.ARCHITECTURE.unique()
):
    if not (
        all(DATA == results.DATA.unique())
        and all(ARCHITECTURE == results.ARCHITECTURE.unique())
    ):
        my_warn(
            f"The hyperparameters specified in {folder_dir} do not match their expected values"
        )
else:
    my_warn(
        f"The hyperparameters specified in {folder_dir} do not match their expected lengths"
    )

assert len(ARCHITECTURE) * len(DATA) == len(results[["ARCHITECTURE", "DATA"]].drop_duplicates())


### ~~~
## ~~~ For each width, from the 4 diffent depths tested with that width, choose the one that has the smallest median validation error, as well as the one that has the smallest validation error overall
### ~~~

mean_results = results.groupby(["width", "depth"]).mean(numeric_only=True)
min_results = results.groupby(["width", "depth"]).min(
    numeric_only=True
)  # ~~~ best results
median_results = results.groupby(["width", "depth"]).median(
    numeric_only=True
)  # ~~~ more typical results

widths = results.width.unique()
WL = []
for w in widths:
    W, L = min_results.query(f"width=={w}").METRIC_rmse_of_median.idxmin()
    #
    # ~~~ Handle duplicates
    if len(WL) > 0:
        if WL[-1] == (W, L):
            W, L = min_results.query(f"width=={w}").METRIC_mae_of_median.idxmin()
        if WL[-1] == (W, L):
            W, L = min_results.query(f"width=={w}").METRIC_max_norm_of_median.idxmin()
        if WL[-1] == (W, L):
            # print(f"Seems L={L} is best for w={w}")
            L = 1 if WL[-1][1] > 1 else 2  # ~~~ for variety
    assert W == w
    WL.append((W, L))

# assert len(set(WL)) == len(WL) == 8, f"Failed to identify 8 different models"
BEST_4_ARCHITECTURES = [f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w, l) in WL]


### ~~~
## ~~~ Diagnostics
### ~~~

def plot(i,title=None):
    plot_trained_model_from_dataframe( results, i, title=f"Model {i}/{len(results)}" if (title is None) else title )

plot( results.METRIC_rmse_of_mean.argmin(), title='"Vanilla" Ensemble with Smallest Validation Loss' )


#
# ~~~ Check for positive correlations
cor_cols = [ _ for _ in results.columns if "cor" in _ ]
min_corr = results[cor_cols].min(axis=1)
plot( min_corr.argmax(), title="Model with Best Min. Corr." )
plot( min_corr.argmin(), title="Model with Worst Min. Corr." )
plot( results.METRIC_median_avg_inverval_score.argmax(), title="Model with Best Interval Score." )
plot( results.METRIC_median_avg_inverval_score.argmin(), title="Model with Worst Interval Score." )



# for i in range(len(results)):
#     print("")
#     print(f"    epochs completed: {results.epochs_completed.iloc[i]}")
#     print(f"    width x depth: {results.width.iloc[i]} x {results.depth.iloc[i]}")
#     print(f"    mse: {results.METRIC_rmse_of_median.iloc[i]}")
#     print(f"    {results.DATA.iloc[i]}")
#     model, x_train_cpu, y_train_cpu, grid, green_curve = plot(i)

#
# ~~~ I went through all 101 fitted models and gave a rough classification of which ones had great/so-so/bad UQ
# for i in range(len(results)): plot(i)
great_uq = [ 0,1,2,3,4,5, 9,10,11, 15,16,17,18,19,20, 33,34,35,36, 88,89,90,91 ]
soso_uq = [ 6,8, 21, 23,24,25,26, 32, 44,45,46,47, 50,51,52,53, 60,61,62,63,64, 77,78,79,80,81,82,83,84,85,86,87, 97, 100 ]
bad_uq = [ 7, 12,13,14, 22, 27,28,29,30,31, 37,38,39,40,41,42,43, 48,49, 54,55,56,57,58,59, 65,66,67,68,69,70,71,72,73,74,75,76, 92,93,94,95,96, 98,99 ]
results.loc[great_uq, "perceived_UQ_quality"] = "great"
results.loc[soso_uq, "perceived_UQ_quality"] = "so-so"
results.loc[bad_uq, "perceived_UQ_quality"] = "bad"
results["perceived_UQ_quality"] = pd.Categorical( # ~~~ re-order from best-to-worst, instead of alphabetical 
    results["perceived_UQ_quality"],
    categories=["great", "so-so", "bad"], # ~~~ any preferred order
    ordered=True
)

#
# ~~~ Try to figure out which metrics indicate what looks to my naked eye like good UQ (interval score fares okayyy; energy score is as bad as coverage; METRIC_quantile_uncertainty_vs_accuracy_slope is the best; METRIC_interpolation_quantile_uncertainty_spread is also good)
metric_cols = [c for c in results.columns if c.startswith("METRIC_")]
metrics = results[metric_cols]
g = results.groupby("perceived_UQ_quality", observed=False)[metric_cols].mean()
for col in g: print(g[col])

for col in metric_cols:
    sns.boxplot(x="perceived_UQ_quality", y=col, data=results)
    plt.title(col)
    plt.show()

from scipy.stats import f_oneway, kruskal
for col in metric_cols:
    groups = [results.loc[results.perceived_UQ_quality == g, col].dropna()
              for g in ["great", "so-so", "bad"]]
    stat, p = f_oneway(*groups)  # parametric
    # stat, p = kruskal(*groups) # nonparametric
    print(f"{col}: p={p:.3g}")


def plot_metrics(metrics_subset, ncols=3):
    nmetrics = len(metrics_subset)
    nrows = math.ceil(nmetrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    for i, col in enumerate(metrics_subset):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sns.boxplot(x="perceived_UQ_quality", y=col, data=results, ax=ax)
        ax.set_title(col)
    # Hide any leftover empty subplots
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()

#
# ~~~ There is a tradeoff between good accuracy and good UQ
acc_cols = [c for c in metric_cols if ("mae" in c or "max_norm" in c or "rmse" in c)]
plot_metrics(acc_cols)

#
# ~~~ My hacky metrics
hack_cols = [c for c in metric_cols if ("_slope" in c or "_cor" in c or "_spread" in c or "_min" in c)]
plot_metrics(hack_cols)

#
# ~~~ The rigorous metrics
rig_cols = [c for c in metric_cols if not (c in acc_cols or c in hack_cols)]
plot_metrics(rig_cols)

