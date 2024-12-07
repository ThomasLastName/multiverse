

from bnns.experiments.paper.univar_det_nn import folder_name
from bnns.utils import load_filtered_json_files



### ~~~
## ~~~ Load the json files from `folder_name` as dictionaries, process them to a format that pandas likes, and combine them into a pandas DataFrame
### ~~~

#
# ~~~ First, remove any lists from the dictionaries, as pandas doesn't like those, before converting to pd.DataFrame 
results = load_filtered_json_files(folder_name)

# #
# # ~~~ Encode the model names, which are strings, as integers for compatibility with certain pandas methods
# model_mapping = {model: idx for idx, model in enumerate(results["MODEL"].unique())}
# results["model_encoded"] = results["MODEL"].map(model_mapping)



### ~~~
## ~~~ Select the results
#### ~~~

mean_results = results.groupby(["DATA"]).mean(numeric_only=True).reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_results, x='n_epochs', y='METRIC_rmse', hue='model_encoded', marker='o')
plt.title('rMSE across Different Models and Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Mean rMSE')
plt.legend(title='Model')
plt.show()
