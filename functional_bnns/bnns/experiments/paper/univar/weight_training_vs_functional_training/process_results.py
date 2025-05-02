
import torch
from matplotlib import pyplot as plt
from importlib import import_module
from bnns.experiments.paper.univar_BBB import folder_name
from bnns.utils import load_filtered_json_files, load_trained_model_from_dataframe, get_attributes_from_row_i, filter_by_attributes, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles
from quality_of_life.my_base_utils import json_to_dict



### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
### ~~~

#
# ~~~ First, remove any lists from the dictionaries, as pandas doesn't like those, before converting to pd.DataFrame 
data = load_filtered_json_files(folder_name)
# data = results[ results.epochs_completed==results.epochs_completed.max() ]
unique_data = data.loc[:,data.nunique()>1]



### ~~~
## ~~~ Prepare plotting utils
### ~~~

def plot_trained_model( dataframe, i, title="Trained Model", n_samples=500 ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    # x_train, y_train, x_test, y_test = data.x_rain, data.y_rain, data.x_test, data.y_test
    grid        =  data.x_test.cpu()
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    plot_predictions = plot_bnn_empirical_quantiles if dataframe.iloc[i].VISUALIZE_DISTRIBUTION_USING_QUANTILES else plot_bnn_mean_and_std
    bnn = load_trained_model_from_dataframe(dataframe,i)
    with torch.no_grad():
        predictions = torch.stack([ bnn(grid) for _ in range(n_samples) ]).squeeze()
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



### ~~~ 
## ~~~ Explore the results
### ~~~

for i in range(len(data)):
    if i>25:
        print("")
        print(f"    i: {i}")
        print("")
        print(unique_data.iloc[i,:7])
        plot_trained_model(data,i)
        print("")
        print("-------------------------------")
        print("")
