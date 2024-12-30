
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module
from bnns.utils import infer_width_and_depth
from quality_of_life.my_base_utils import json_to_dict, my_warn, print_dict

from bnns.experiments.paper.univar_det_nn import folder_name, DATA, ARCHITECTURE, LR



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
        and len(ARCHITECTURE) == 24 == len(results.MODEL.unique())
        and len(LR) == 5 == len(results.LR.unique())
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
## ~~~ For each width, from the 4 diffent depths tested with that width, choose the one that has the smallest median validation error, as well as the one that has the smallest validation error overall
### ~~~

mean_results = results.groupby(["width","depth"]).mean(numeric_only=True)
min_results = results.groupby(["width","depth"]).min(numeric_only=True)         # ~~~ best results
median_results = results.groupby(["width","depth"]).median(numeric_only=True)   # ~~~ more typical results

widths = results.width.unique()
WL = []
for w in widths:
    for df in (min_results,median_results):
        W,L = df.query(f"width=={w}").METRIC_rmse.idxmin()
        #
        # ~~~ Handle duplicates
        if len(WL)>0:
            if WL[-1]==(W,L):
                W,L = df.query(f"width=={w}").METRIC_mae.idxmin()
            if WL[-1]==(W,L):
                W,L = df.query(f"width=={w}").METRIC_max_norm.idxmin()
            if WL[-1]==(W,L):
                # print(f"Seems L={L} is best for w={w}")
                L = 1 if WL[-1][1]>1 else 2 # ~~~ for variety
        assert W==w                
        WL.append((W,L))

assert len(set(WL))==len(WL)==12, f"Failed to identify 12 different models"
BEST_12_ARCHITECTURES = [ f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w,l) in WL ]

if __name__=="__main__":
    #
    # ~~~ "Trim the fat" from a dataframe by saving only the listed columns
    columns_to_save = [ "width", "depth", "METRIC_rmse", "METRIC_mae", "METRIC_max_norm" ]  # ~~~ annoyingly, throws an error if made a tuple instead of a list
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
