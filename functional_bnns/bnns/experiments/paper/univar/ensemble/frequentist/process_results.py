
import os
import pandas as pd
from bnns.experiments.paper.univar.ensemble.frequentist import folder_name

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
