
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bnns.utils import infer_width_and_depth, plot_trained_model_from_dataframe
from quality_of_life.my_base_utils import my_warn

from bnns.experiments.paper.univar.ensemble.frequentist import folder_name, DATA, ARCHITECTURE, LR



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
## ~~~ Process the slightly more
### ~~~

results = infer_width_and_depth(results)

#
# ~~~ Verify that DATA==results.DATA.unique(), ARCHITECTURE==results.ARCHITECTURE.unique(), and LR==results.LR.unique()
if (
        len(DATA) == 2 == len(results.DATA.unique())
        and len(ARCHITECTURE) == 16 == len(results.ARCHITECTURE.unique())
    ):
    if not (
                all(DATA==results.DATA.unique())
                and all(ARCHITECTURE==results.ARCHITECTURE.unique())
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
        W,L = df.query(f"width=={w}").METRIC_rmse_of_median.idxmin()
        #
        # ~~~ Handle duplicates
        if len(WL)>0:
            if WL[-1]==(W,L):
                W,L = df.query(f"width=={w}").METRIC_mae_of_median.idxmin()
            if WL[-1]==(W,L):
                W,L = df.query(f"width=={w}").METRIC_max_norm_of_median.idxmin()
            if WL[-1]==(W,L):
                # print(f"Seems L={L} is best for w={w}")
                L = 1 if WL[-1][1]>1 else 2 # ~~~ for variety
        assert W==w
        WL.append((W,L))

assert len(set(WL))==len(WL)==8, f"Failed to identify 8 different models"
BEST_8_ARCHITECTURES = [ f"univar_NN.univar_NN_{'_'.join(l*[str(w)])}" for (w,l) in WL ]

for i in range(len(results)):
    print("")
    print(f"    epochs completed: {results.epochs_completed.iloc[i]}")
    print(f"    width x depth: {results.width.iloc[i]} x {results.depth.iloc[i]}")
    print(f"    mse: {results.METRIC_rmse_of_median.iloc[i]}")
    plot_trained_model_from_dataframe( results, i, title=f"Model {i}/{len(results)}... epochs: {results.epochs_completed.iloc[i]}, width x depth: {results.width.iloc[i]} x {results.depth.iloc[i]}, mse: {results.METRIC_rmse_of_median.iloc[i]:.2f}" )
