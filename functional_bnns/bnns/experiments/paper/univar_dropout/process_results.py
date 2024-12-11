
import numpy as np
import torch
from matplotlib import pyplot as plt
from importlib import import_module
from bnns.experiments.paper.univar_dropout import folder_name
from bnns.utils import load_filtered_json_files, load_trained_model_from_dataframe, get_attributes_from_row_i, filter_by_attributes, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles
from quality_of_life.my_base_utils import json_to_dict



### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
### ~~~

#
# ~~~ First, remove any lists from the dictionaries, as pandas doesn't like those, before converting to pd.DataFrame 
results = load_filtered_json_files(folder_name)
unique_results = results.loc[:,results.nunique()>1]

#
# ~~~ Average over all data train/val folds
mean_results = unique_results.groupby(["MODEL","LR","n_epochs"]).mean(numeric_only=True).reset_index()

#
# ~~~ Sanity check that `groupby` works as intended
model, lr, n = get_attributes_from_row_i( mean_results, 0, "MODEL", "LR", "n_epochs" )
filtered_results = filter_by_attributes( unique_results, MODEL=model, LR=lr, n_epochs=n )
assert filtered_results.shape == (2,30)
a = filtered_results.mean(numeric_only=True).to_numpy()
b = mean_results.iloc[0,1:].to_numpy()
assert np.array_equal(a,b)



### ~~~
## ~~~ Choose the "best" hyperparameters
### ~~~

acceptable_results = mean_results[
       ( mean_results.METRIC_uncertainty_vs_accuracy_slope_quantile > 0 ) &
       ( mean_results.METRIC_uncertainty_vs_accuracy_cor_quantile > 0 ) &
       ( mean_results.METRIC_extrapolation_uncertainty_vs_proximity_slope_quantile > 0 ) &
       ( mean_results.METRIC_uncertainty_vs_proximity_cor_quantile > 0 ) &
       ( mean_results.METRIC_interpolation_uncertainty_vs_proximity_slope_quantile > 0 ) &
       ( mean_results.METRIC_uncertainty_vs_accuracy_slope_pm2_std > 0 ) &
       ( mean_results.METRIC_uncertainty_vs_accuracy_cor_pm2_std > 0 ) &
       ( mean_results.METRIC_extrapolation_uncertainty_vs_proximity_slope_pm2_std > 0 ) &
       ( mean_results.METRIC_uncertainty_vs_proximity_cor_pm2_std > 0 ) &
       ( mean_results.METRIC_interpolation_uncertainty_vs_proximity_slope_pm2_std > 0 )
]
top_25_percent_by_loss = mean_results[ mean_results.METRIC_rmse_of_mean <= mean_results.METRIC_rmse_of_mean.quantile(q=0.25) ]
best_UQ = acceptable_results.METRIC_interpolation_uncertainty_spread_pm2_std.argmax()


### ~~~
## ~~~ Prepare plotting utils
### ~~~

def plot_trained_model( dataframe, i, title="Trained Model" ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    # x_train, y_train, x_test, y_test = data.x_rain, data.y_rain, data.x_test, data.y_test
    grid        =  data.x_test.cpu()
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    plot_predictions = plot_bnn_empirical_quantiles if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
    nn = load_trained_model_from_dataframe(dataframe,i)
    with torch.no_grad():
        predictions = torch.stack([ nn(grid) for _ in range(1500) ]).squeeze()
        fig, ax = plt.subplots(figsize=(12,6))
        fig, ax = plot_predictions(
            fig = fig,
            ax = ax,
            grid = grid,
            green_curve = green_curve,
            x_train = x_train_cpu,
            y_train = y_train_cpu,
            predictions = predictions,
            extra_std = 0.,
            how_many_individual_predictions = 0,
            title = title
            )
        plt.show()

best_UQ = acceptable_results.METRIC_interpolation_uncertainty_spread_pm2_std.argmax()
model, lr, n = get_attributes_from_row_i( acceptable_results, best_UQ, "MODEL", "LR", "n_epochs" )
good_models = filter_by_attributes( results, MODEL=model, LR=lr, n_epochs=n )
plot_trained_model( good_models, 0, title="Model with the best UQ amongst acceptable results" )

best_loss = acceptable_results.METRIC_rmse_of_mean.argmin()
model, lr, n = get_attributes_from_row_i( acceptable_results, best_loss, "MODEL", "LR", "n_epochs" )
good_models = filter_by_attributes( results, MODEL=model, LR=lr, n_epochs=n )
plot_trained_model( good_models, 0, title="Model with the best loss amongst acceptable results" )

plot_trained_model( results, results.METRIC_rmse_of_mean.argmin(), title="Model with the best loss amongst all results" )
plot_trained_model( results, results.METRIC_uncertainty_vs_accuracy_cor_quantile.argmax(), title="Model with the best UQ amongst all results" )

"""
 - full loss
 - average predictive error
 - interval score
"""


### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
### ~~~


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_results, x='METRIC_rmse_of_mean', y='METRIC_interpolation_uncertainty_spread_pm2_std', hue='MODEL', marker='o')
plt.title('rMSE across Different Models and Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Mean rMSE')
# plt.legend(title='Model')
plt.show()
