import os
import numpy as np
import pandas as pd
import torch
from bnns.experiments.paper.univar.weight_priors import folder_name
from bnns.utils import load_filtered_json_files, plot_trained_model_from_dataframe, add_metrics, print_dict


### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
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


results = add_metrics(results)
unique_data = results.loc[:, results.nunique() > 1] # ~~~ drop any column where the number of unique values in that column is 1 (e.g., "DEVICE", assuming all experiments were run on the same device)
# data = results[ results.epochs_completed==results.epochs_completed.max() ]


### ~~~
## ~~~ Diagnostics
### ~~~

metric_cols = [c for c in results.columns if c.startswith("METRIC_")]
metrics = results[metric_cols]
z = (metrics - metrics.mean()) / metrics.std()
print(z.loc[[31, 32]].T.sort_values(by=31, key=np.abs, ascending=False).head(6))



def plot(i, title=None, val=False, extra_std=True, **kwargs):
    return plot_trained_model_from_dataframe( results, i, val=val, title=f"Model {i}/{len(results)}" if (title is None) else title, extra_std=results.LIKELIHOOD_STD.iloc[i] if extra_std else 0, **kwargs )


# for i in range(len(results)):
#     if results.METRIC_rmse_of_mean.iloc[i] < 0.04:
#         model, x_train_cpu, y_train_cpu, grid, green_curve = plot(i)


plot( results.METRIC_rmse_of_mean.argmin(), title="Model with Best Validation Accuracy" )
plot( results.METRIC_val_lik.argmax(), title="Model with Best Validation Likelihood" )
plot( results.METRIC_val_loss.argmin(), title="Model with Smallest Validation Loss" )
plot( abs(results.METRIC_95_coverage - 0.95).argmin(), title="Model with Nearly Perfect 95% Coverage", val=True )
# plot( results.METRIC_kl_div.argmin(), title="Model with Smallest KL Divergence" )
plot( results.METRIC_median_avg_inverval_score.argmax(), title="Model with Best Interval Score." )
plot( results.METRIC_median_avg_inverval_score.argmin(), title="Model with Worst Interval Score." )


#
# ~~~ Check for positive correlations
cor_cols = [ _ for _ in results.columns if "_cor" in _ ]
min_corr = results[cor_cols].min(axis=1)

plot( min_corr.argmax(), title="Model with Best Min. Corr." )
plot( min_corr.argmin(), title="Model with Worst Min. Corr." )



# for i in range(len(results)):
#     print("")
#     print(f"    epochs completed: {results.epochs_completed.iloc[i]}")
#     print(f"    width x depth: {results.width.iloc[i]} x {results.depth.iloc[i]}")
#     print(f"    mse: {results.METRIC_rmse_of_median.iloc[i]}")
#     print(f"    {results.DATA.iloc[i]}")
#     model, x_train_cpu, y_train_cpu, grid, green_curve = plot(i)


# ### ~~~
# ## ~~~ Observe that the only models which fail to converge are the ones for which the likehood std. is large (this makes sense)
# ### ~~~

# plt.figure(figsize=(6, 4))
# sns.scatterplot(
#     data=unique_results, x="LIKELIHOOD_STD", y="METRIC_rmse_of_mean", alpha=0.6
# )
# plt.xscale("log")
# plt.yscale("log")
# plt.title("Validation Accuracy (Lower is Better) vs. Likelihood Standard Deviation")
# plt.xlabel("LIKELIHOOD_STD (log scale)")
# plt.ylabel("rMSE of Posterior Predictive Mean (log scale)")
# plt.show()


# ### ~~~
# ## ~~~ Check out some models
# ### ~~~

# indices_with_great_accuracy_despite_high_noise = np.where(
#     (results.LIKELIHOOD_STD == 0.1) & (results.METRIC_rmse_of_mean < 0.02)
# )[0]
# for i in indices_with_great_accuracy_despite_high_noise[:3]:
#     plot_trained_model_from_dataframe(results, i, title="A Model with Coverage {}")


# def plot_k(dataframe, metric, small=True, k=3):
#     selector = k_smallest_indices if small else k_largest_indices
#     indices = selector(dataframe, "METRIC_" + metric, k=k)
#     for i in indices:
#         plot_trained_model_from_dataframe(
#             results, i, title=f"A Model with {metric} {results.METRIC_coverage.iloc[i]}"
#         )


# plot_k(results, "coverage", small=False)  # ~~~ plot the 3 models with highest coverage
# plot_k(
#     abs(results.METRIC_coverage - 0.95), "coverage", small=True
# )  # ~~~ plot the 3 models coverage closest to 95%
# plot_k(
#     results, "median_avg_inverval_score", small=True
# )  # ~~~ plot the 3 models with best interval score
# plot_k(
#     results, "median_energy_score", small=True
# )  # ~~~ plot the 3 models with best energy score


# ### ~~~
# ## ~~~ For each width, from the 2 diffent depths tested with that width, choose the depth that has the smallest median validation error
# ### ~~~

# mean_results = results.groupby(["width","depth"]).mean(numeric_only=True)
# min_results = results.groupby(["width","depth"]).min(numeric_only=True)         # ~~~ best results
# median_results = results.groupby(["width","depth"]).median(numeric_only=True)   # ~~~ more typical results

# widths = results.width.unique()
# WL = []
# for w in widths:
#     W,L = median_results.query(f"width=={w}").METRIC_rmse_of_mean.idxmin()
#     assert W==w
#     WL.append((W,L))

# assert len(set(WL))==len(WL)==4, f"Failed to identify 12 different models"
# BEST_4_ARCHITECTURES = [ f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w,l) in WL ]


# if __name__=="__main__":
#     #
#     # ~~~ "Trim the fat" from a dataframe by saving only the listed columns
#     columns_to_save = [ "width", "depth", "METRIC_rmse_of_median", "METRIC_mae_of_median", "METRIC_max_norm_of_median" ]
#     trim = lambda df: df.reset_index()[columns_to_save].round(3).to_string(index=False)     # ~~~ reset the index, so that "width" and "depth" are restored to being columns (as opposed to being the index)
#     #
#     # ~~~ Print the average resutls by width and depth
#     print("")
#     print("    Average Resutls (across all other hyperparameters) by Width and Depth")
#     print("")
#     print(trim(mean_results))
#     #
#     # ~~~ Print the best resutls by width and depth
#     print("")
#     print("    Best Resutls (across all other hyperparameters) by Width and Depth")
#     print("")
#     print(trim(min_results))
#     #
#     # ~~~ Plot a model or two, as a sanity check
#     def plot(criterion):
#         plt.figure(figsize=(12,6))
#         sns.lineplot(
#                 data = results,
#                 x = "width",
#                 y = "METRIC_rmse",
#                 hue = "depth",
#                 marker = "o",
#                 estimator = criterion,
#                 errorbar = ("pi",95) if criterion=="median" else ("sd",2)
#             )
#         plt.title("Validation rMSE in Various Experiments by Model Width and Depth")
#         plt.xlabel("Width")
#         plt.ylabel(f"{criterion} rMSE")
#         plt.legend(title="Depth")
#         plt.show()


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




# ### ~~~
# ## ~~~ For each width, from the 2 diffent depths tested with that width, choose the depth that has the smallest median validation error
# ### ~~~

# mean_results = results.groupby(["width","depth"]).mean(numeric_only=True)
# min_results = results.groupby(["width","depth"]).min(numeric_only=True)         # ~~~ best results
# median_results = results.groupby(["width","depth"]).median(numeric_only=True)   # ~~~ more typical results

# widths = results.width.unique()
# WL = []
# for w in widths:
#     W,L = median_results.query(f"width=={w}").METRIC_rmse_of_mean.idxmin()
#     assert W==w
#     WL.append((W,L))

# assert len(set(WL))==len(WL)==4, f"Failed to identify 12 different models"
# BEST_4_ARCHITECTURES = [ f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w,l) in WL ]


# if __name__=="__main__":
#     #
#     # ~~~ "Trim the fat" from a dataframe by saving only the listed columns
#     columns_to_save = [ "width", "depth", "METRIC_rmse_of_median", "METRIC_mae_of_median", "METRIC_max_norm_of_median" ]
#     trim = lambda df: df.reset_index()[columns_to_save].round(3).to_string(index=False)     # ~~~ reset the index, so that "width" and "depth" are restored to being columns (as opposed to being the index)
#     #
#     # ~~~ Print the average resutls by width and depth
#     print("")
#     print("    Average Resutls (across all other hyperparameters) by Width and Depth")
#     print("")
#     print(trim(mean_results))
#     #
#     # ~~~ Print the best resutls by width and depth
#     print("")
#     print("    Best Resutls (across all other hyperparameters) by Width and Depth")
#     print("")
#     print(trim(min_results))
#     #
#     # ~~~ Plot a model or two, as a sanity check
#     def plot(criterion):
#         plt.figure(figsize=(12,6))
#         sns.lineplot(
#                 data = results,
#                 x = "width",
#                 y = "METRIC_rmse",
#                 hue = "depth",
#                 marker = "o",
#                 estimator = criterion,
#                 errorbar = ("pi",95) if criterion=="median" else ("sd",2)
#             )
#         plt.title("Validation rMSE in Various Experiments by Model Width and Depth")
#         plt.xlabel("Width")
#         plt.ylabel(f"{criterion} rMSE")
#         plt.legend(title="Depth")
#         plt.show()


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
