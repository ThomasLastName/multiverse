
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module
from bnns.utils import infer_width_and_depth
from quality_of_life.my_base_utils import json_to_dict, my_warn, print_dict

from bnns.experiments.paper.univar_dropout import folder_name, DATA, ARCHITECTURE, LR



### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, processed in a format that pandas likes (remove any lists), and combined into a pandas DataFrame
### ~~~

folder_dir = os.path.split(folder_name)[0]
try:
    results = pd.read_csv(os.path.join( folder_dir, "results.csv" ))
except FileNotFoundError:
    print("")
    print("    Processing the raw results and storing them in .csv form (this should only need to be done once).")
    print("")
    from bnns.utils import load_filtered_json_files
    results = load_filtered_json_files(folder_name)
    results.to_csv(os.path.join( folder_dir, "results.csv" ))
except:
    raise



### ~~~
## ~~~ Process the dataframe slightly
### ~~~

results = infer_width_and_depth(results)

#
# ~~~ Verify that DATA==results.DATA.unique(), ARCHITECTURE==results.MODEL.unique(), and LR==results.LR.unique()
if (
        len(DATA) == 2 == len(results.DATA.unique())
        and len(ARCHITECTURE) == 12 == len(results.MODEL.unique())
        and len(LR) == 4 == len(results.LR.unique())
    ):
    if not (
                all(DATA==results.DATA.unique())
                and all(ARCHITECTURE==results.MODEL.unique())
                and all(LR==results.LR.unique())
            ):
        my_warn(f"The hyperparameters specified in {folder_dir} do not match their expected values")
else:
    my_warn(f"The hyperparameters specified in {folder_dir} do not match their expected lengths")



### ~~~
## ~~~ For each width, from the 2 diffent depths tested with that width, choose the depth that has the smallest median validation error, as well as the one that has the smallest validation error overall
### ~~~

mean_results = results.groupby(["width","depth"]).mean(numeric_only=True)
min_results = results.groupby(["width","depth"]).min(numeric_only=True)         # ~~~ best results
median_results = results.groupby(["width","depth"]).median(numeric_only=True)   # ~~~ more typical results

widths = results.width.unique()
WL = []
for w in widths:
    W,L = median_results.query(f"width=={w}").METRIC_rmse_of_mean.idxmin()
    assert W==w
    WL.append((W,L))

assert len(set(WL))==len(WL)==6, f"Failed to identify 12 different models"
BEST_6_ARCHITECTURES = [ f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w,l) in WL ]


if __name__=="__main__":
    #
    # ~~~ "Trim the fat" from a dataframe by saving only the listed columns
    columns_to_save = [ "width", "depth", "METRIC_rmse_of_median", "METRIC_mae_of_median", "METRIC_max_norm_of_median" ]
    trim = lambda df: df.reset_index()[columns_to_save].round(3).to_string(index=False)     # ~~~ reset the index, so that "width" and "depth" are restored to being columns (as opposed to being the index)
    #
    # ~~~ Print the average resutls by width and depth
    print("")
    print("    Average Resutls (across all other hyperparameters) by Width and Depth")
    print("")
    print(trim(mean_results))
    #
    # ~~~ Print the best resutls by width and depth
    print("")
    print("    Best Resutls (across all other hyperparameters) by Width and Depth")
    print("")
    print(trim(min_results))
    #
    # ~~~ Plot a model or two, as a sanity check
    try:
        import seaborn as sns
        def plot(criterion):
            plt.figure(figsize=(12,6))
            sns.lineplot(
                    data = results,
                    x = "width",
                    y = "METRIC_rmse",
                    hue = "depth",
                    marker = "o",
                    estimator = criterion,
                    errorbar = ("pi",95) if criterion=="median" else ("sd",2)
                )
            plt.title("Validation rMSE in Various Experiments by Model Width and Depth")
            plt.xlabel("Width")
            plt.ylabel(f"{criterion} rMSE")
            plt.legend(title="Depth")
            plt.show()
    except ModuleNotFoundError:
        pass
    except:
        raise

# ### ~~~
# ## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
# ### ~~~

# #
# # ~~~ First, remove any lists from the dictionaries, as pandas doesn't like those, before converting to pd.DataFrame 
# results = load_filtered_json_files(folder_name)
# unique_results = results.loc[:,results.nunique()>1]

# #
# # ~~~ Average over all data train/val folds
# mean_results = unique_results.groupby(["MODEL","LR","n_epochs"]).mean(numeric_only=True).reset_index()

# #
# # ~~~ Sanity check that `groupby` works as intended
# model, lr, n = get_attributes_from_row_i( mean_results, 0, "MODEL", "LR", "n_epochs" )
# filtered_results = filter_by_attributes( unique_results, MODEL=model, LR=lr, n_epochs=n )
# assert filtered_results.shape == (2,30)
# a = filtered_results.mean(numeric_only=True).to_numpy()
# b = mean_results.iloc[0,1:].to_numpy()
# assert np.array_equal(a,b)



# ### ~~~
# ## ~~~ Choose the "best" hyperparameters
# ### ~~~

# acceptable_results = mean_results[
#        ( mean_results.METRIC_uncertainty_vs_accuracy_slope_quantile > 0 ) &
#        ( mean_results.METRIC_uncertainty_vs_accuracy_cor_quantile > 0 ) &
#        ( mean_results.METRIC_extrapolation_uncertainty_vs_proximity_slope_quantile > 0 ) &
#        ( mean_results.METRIC_uncertainty_vs_proximity_cor_quantile > 0 ) &
#        ( mean_results.METRIC_interpolation_uncertainty_vs_proximity_slope_quantile > 0 ) &
#        ( mean_results.METRIC_uncertainty_vs_accuracy_slope_pm2_std > 0 ) &
#        ( mean_results.METRIC_uncertainty_vs_accuracy_cor_pm2_std > 0 ) &
#        ( mean_results.METRIC_extrapolation_uncertainty_vs_proximity_slope_pm2_std > 0 ) &
#        ( mean_results.METRIC_uncertainty_vs_proximity_cor_pm2_std > 0 ) &
#        ( mean_results.METRIC_interpolation_uncertainty_vs_proximity_slope_pm2_std > 0 )
# ]
# top_25_percent_by_loss = mean_results[ mean_results.METRIC_rmse_of_mean <= mean_results.METRIC_rmse_of_mean.quantile(q=0.25) ]
# best_UQ = acceptable_results.METRIC_interpolation_uncertainty_spread_pm2_std.argmax()


# ### ~~~
# ## ~~~ Prepare plotting utils
# ### ~~~

# def plot_trained_model( dataframe, i, title="Trained Model" ):
#     data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
#     # x_train, y_train, x_test, y_test = data.x_rain, data.y_rain, data.x_test, data.y_test
#     grid        =  data.x_test.cpu()
#     green_curve =  data.y_test.cpu().squeeze()
#     x_train_cpu = data.x_train.cpu()
#     y_train_cpu = data.y_train.cpu().squeeze()
#     plot_predictions = plot_bnn_empirical_quantiles if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
#     nn = load_trained_model_from_dataframe(dataframe,i)
#     with torch.no_grad():
#         predictions = torch.stack([ nn(grid) for _ in range(1500) ]).squeeze()
#         fig, ax = plt.subplots(figsize=(12,6))
#         fig, ax = plot_predictions(
#             fig = fig,
#             ax = ax,
#             grid = grid,
#             green_curve = green_curve,
#             x_train = x_train_cpu,
#             y_train = y_train_cpu,
#             predictions = predictions,
#             extra_std = 0.,
#             how_many_individual_predictions = 0,
#             title = title
#             )
#         plt.show()

# best_UQ = acceptable_results.METRIC_interpolation_uncertainty_spread_pm2_std.argmax()
# model, lr, n = get_attributes_from_row_i( acceptable_results, best_UQ, "MODEL", "LR", "n_epochs" )
# good_models = filter_by_attributes( results, MODEL=model, LR=lr, n_epochs=n )
# plot_trained_model( good_models, 0, title="Model with the best UQ amongst acceptable results" )

# best_loss = acceptable_results.METRIC_rmse_of_mean.argmin()
# model, lr, n = get_attributes_from_row_i( acceptable_results, best_loss, "MODEL", "LR", "n_epochs" )
# good_models = filter_by_attributes( results, MODEL=model, LR=lr, n_epochs=n )
# plot_trained_model( good_models, 0, title="Model with the best loss amongst acceptable results" )

# plot_trained_model( results, results.METRIC_rmse_of_mean.argmin(), title="Model with the best loss amongst all results" )
# plot_trained_model( results, results.METRIC_uncertainty_vs_accuracy_cor_quantile.argmax(), title="Model with the best UQ amongst all results" )

# """
#  - full loss
#  - average predictive error
#  - interval score
# """


# ### ~~~
# ## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
# ### ~~~


# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# sns.lineplot(data=mean_results, x='METRIC_rmse_of_mean', y='METRIC_interpolation_uncertainty_spread_pm2_std', hue='MODEL', marker='o')
# plt.title('rMSE across Different Models and Epochs')
# # plt.xlabel('Number of Epochs')
# # plt.ylabel('Mean rMSE')
# # plt.legend(title='Model')
# plt.show()
