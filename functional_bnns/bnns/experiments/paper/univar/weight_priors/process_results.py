import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from importlib import import_module
from bnns.experiments.paper.univar.weight_training_vs_functional_training import (
    folder_name,
)
from bnns.utils import (
    load_filtered_json_files,
    load_trained_model_from_dataframe,
    plot_bnn_mean_and_std,
    plot_bnn_empirical_quantiles,
)


### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
### ~~~


folder_dir = os.path.split(folder_name)[0]
try:
    results = pd.read_csv(os.path.join(folder_dir, "results.csv"))
except FileNotFoundError:
    print("")
    print(
        "    Processing the raw results and storing them in .csv form (this should only need to be done once)."
    )
    print("")
    from bnns.utils import load_filtered_json_files

    results = load_filtered_json_files(folder_name)
    results.to_csv(os.path.join(folder_dir, "results.csv"))
except:
    raise

# data = results[ results.epochs_completed==results.epochs_completed.max() ]
unique_data = results.loc[:, results.nunique() > 1]


### ~~~
## ~~~ Prepare plotting utils
### ~~~


def plot_trained_model(dataframe, i, title="Trained Model", n_samples=100):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    grid = data.x_test.cpu()
    green_curve = data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    plot_predictions = (
        plot_bnn_empirical_quantiles
        if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES
        else plot_bnn_mean_and_std
    )
    bnn = load_trained_model_from_dataframe(dataframe, i)
    with torch.no_grad():
        predictions = bnn(grid, n=n_samples).squeeze()
        fig, ax = plt.subplots(figsize=(12, 6))
        fig, ax = plot_predictions(
            fig=fig,
            ax=ax,
            grid=grid,
            green_curve=green_curve,
            x_train=x_train_cpu,
            y_train=y_train_cpu,
            predictions=predictions,
            extra_std=0.0,
            how_many_individual_predictions=0,
            title=title,
        )
        plt.show()
    return bnn


### ~~~
## ~~~ Explore the results
### ~~~

for i in range(len(results)):
    if True:
        print("")
        print(f"    i: {i}")
        print("")
        print(results.iloc[i, :7])
        bnn = plot_trained_model(results, i)
        print("")
        print("-------------------------------")
        print("")
