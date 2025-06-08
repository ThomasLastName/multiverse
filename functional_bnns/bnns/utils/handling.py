
import numpy as np
import pandas as pd
import torch

import os
import pytz
import argparse
from tqdm import tqdm
from glob import glob
from datetime import datetime
from importlib import import_module
from quality_of_life.ansi import bcolors
from quality_of_life.my_base_utils import process_for_saving, my_warn, json_to_dict, support_for_progress_bars



### ~~~
## ~~~ Non-math non-plotting stuff (e.g., data processing)
### ~~~

#
# ~~~ Flatten and concatenate all the parameters in a model
flatten_parameters = lambda model: torch.cat([ p.view(-1) for p in model.parameters() ])

#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def convert_to_list_and_check_items( x, classes, other_requirement=lambda*args:True ):
    #
    # ~~~ Convert x to a list (if x is already a list, this has no effect)
    try: X = list(x)
    except TypeError: X = [x]
    except: raise
    assert isinstance(X,list), "Unexpected error: both list(x) and [x] failed to create a list out of x."
    #
    # ~~~ Verify the type of each item in the list
    for item in X:
        assert isinstance(item,classes)
        assert other_requirement(item), "The user supplied check was not satisfied."
    #
    # ~~~ Return the list whose items all meet the type and other requirements
    return X

#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def non_negative_list( x, integer_only=False ):
    return convert_to_list_and_check_items(
            x = x,
            classes = int if integer_only else (int,float),
            other_requirement = lambda item: item>=0
        )

#
# ~~~ A standard early stopping rule (https://stackoverflow.com/a/73704579/11595884)
class EarlyStopper:
    def __init__( self, patience=20, delta=0.05 ):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_count = 0
        self.min_val_loss = float('inf')
    def __call__(self,val_loss):
        #
        # ~~~ If the validation loss is decreasing, reset the counter
        if val_loss <= self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        #
        # ~~~ Compute the relative difference between the current `val_loss` and smallest yet `self.min_val_loss`
        if self.min_val_loss>0:
            rel_val_loss = val_loss/self.min_val_loss - 1   # ~~~ `== ( val_loss - self.min_val_loss ) / self.min_val_loss` but more numerically stable
        else:
            rel_val_loss = abs(val_loss-self.min_val_loss) / abs(self.min_val_loss)
        #
        # ~~~ If the current val_loss is "more than delta worse" relative to min_val_loss, increment the counter
        if rel_val_loss > self.delta:
            self.counter += 1
            self.max_count = max( self.counter, self.max_count )
        #
        # ~~~ Stop iff max_count >= patience
        if self.max_count >= self.patience:
            return True
        else:
            return False

#
# ~~~ Load all the .json files in a directory to data frame
def load_filtered_json_files( directory, verbose=True ):
    #
    # ~~~ Load (as a list of dictionaries) all the .json files in a directory that don't start with "RUN_THIS"
    with support_for_progress_bars():
        json_files = glob(os.path.join(directory,'*.json'))
        json_files = [ json for json in json_files if not os.path.split(json)[1].startswith("RUN_THIS") ]
        all_dicts  = [ json_to_dict(json) for json in ( tqdm(json_files,desc="Loading json files") if verbose else json_files ) ]
    #
    # ~~~ Remove from each dictionary any key/value pair where the value is a list, as pandas doesn't like those
    all_filtered_dicts = [
            {
                k:v
                for k,v in dict.items()
                if not isinstance(v,list)
            }
            for dict in all_dicts
        ]
    return pd.DataFrame(all_filtered_dicts)

#
# ~~~ Infer the width of each model
def infer_width_and_depth( dataframe, field="ARCHITECTURE" ):
    #
    # ~~~ Infer the width of each model
    width_mapping = {}
    for model in dataframe[field].unique():
        text_after_last_underscore = model[model.rfind("_")+1:] # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        width_mapping[model] = int(text_after_last_underscore)
    dataframe["width"] = dataframe[field].map(width_mapping)
    #
    # ~~~ Infer the depth of each model
    depth_mapping = {}
    for model in dataframe[field].unique():
        how_many_underscores = len(model.split("_"))-1 # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        depth_mapping[model] = how_many_underscores-2 if how_many_underscores>1 else 2
    dataframe["depth"] = dataframe[field].map(depth_mapping)
    return dataframe

#
# ~~~ Get the dataframe.iloc[i,"arg"] for all the arguments `args`
def get_attributes_from_row_i(dataframe,i,*args):
    return [
            dataframe.iloc[i][arg]
            for arg in args
        ]

#
# ~~~ Filter a dataframe by multiple attributes
def filter_by_attributes(dataframe,**kwargs):
    filtered_results = dataframe
    for key, value in kwargs.items():
        filtered_results = filtered_results[ filtered_results[key]==value ]
    return filtered_results

#
# ~~~ Load a trained BNN, based on the string `architecture` that points to the file where the model is defined
def load_trained_bnn( architecture:str, model:str, state_dict_path ):
    #
    # ~~~ Load the untrained model
    import bnns
    architecture = import_module(f"bnns.models.{architecture}").NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    model = getattr(bnns,model)(*architecture)
    model.load_state_dict(torch.load(state_dict_path))
    return model

#
# ~~~ Load a trained ensemble, based on the string `architecture` that points to the file where the model is defined
def load_trained_ensemble( architecture:str, n_models:str, state_dict_path ):
    #
    # ~~~ Load the untrained model
    import bnns
    architecture = import_module(f"bnns.models.{architecture}").NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    ensemble = bnns.Ensemble.SequentialSteinEnsemble( architecture, n_models )
    ensemble.load_state_dict(torch.load(state_dict_path))
    return ensemble

#
# ~~~ Load a trained conventional neural network, based on the dataframe of results you get from hyperparameter search
def load_trained_nn( architecture:str, state_dict_path ):
    import bnns
    architecture = import_module(f"bnns.models.{architecture}").NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    architecture.load_state_dict(torch.load(state_dict_path))
    return architecture

#
# ~~~ Load a trained model, based on the dataframe of results you get from hyperparameter search
def load_trained_model_from_dataframe( results_dataframe, i ):
    #
    # ~~~ Load the untrained model
    architecture = results_dataframe.iloc[i].ARCHITECTURE
    state_dict_path = results_dataframe.iloc[i].STATE_DICT_PATH
    try:
        model = results_dataframe.iloc[i].MODEL
        return load_trained_bnn( architecture, model, state_dict_path )
    except:
        try:
            n_models = results_dataframe.iloc[i].N_MODELS
            return load_trained_ensemble( architecture, n_models, state_dict_path )
        except:
            return load_trained_nn( architecture, state_dict_path )

#
# ~~~ Generate a .json filename based on the current datetime
def generate_json_filename(verbose=True,message=None):
    #
    # ~~~ Generate a .json filename
    time = datetime.now(pytz.timezone('US/Central'))        # ~~~ current date and time CST
    file_name = str(time)
    file_name = file_name[:file_name.find(".")]             # ~~~ remove the number of milliseconds (indicated with ".") 
    file_name = file_name.replace(" ","_").replace(":","-") # ~~~ replace blank space (between date and time) with an underscore and colons (hr:mm:ss) with dashes
    file_name = process_for_saving(file_name+".json")       # ~~~ procsess_for_saving("path_that_exists.json") returns "path_that_exists (1).json"
    #
    # ~~~ Craft a message to print
    if verbose:
        if time.hour > 12:
            hour = time.hour - 12
            suffix = "pm"
        else:
            hour = time.hour
            suffix = "am"
        base_message = bcolors.OKBLUE + f"    Generating file name {file_name} at {hour}:{time.minute:02d}{suffix} CST" + bcolors.HEADER
        if message is not None:
            if not message[0]==" ":
                message = " " + message
            base_message += message
        print(base_message)
    return file_name

#
# ~~~ My version of the missing feature: a `dataset.to` method
def set_Dataset_attributes( dataset, device, dtype ):
    try:
        #
        # ~~~ Directly access and modify the underlying tensors
        dataset.X = dataset.X.to( device=device, dtype=dtype )
        dataset.y = dataset.y.to( device=device, dtype=dtype )
        return dataset
    except AttributeError:
        #
        # ~~~ Redefine the __getattr__ method (this is hacky; I don't know a better way; also, chat-gpt proposed this)
        class ModifiedDataset(torch.utils.data.Dataset):
            def __init__(self,original_dataset):
                self.original_dataset = original_dataset
                self.device = device
                self.dtype = dtype
            def __getitem__(self,index):
                x, y = self.original_dataset[index]
                return x.to( device=self.device, dtype=self.dtype ), y.to( device=self.device, dtype=self.dtype )
            def __len__(self):
                return len(self.original_dataset)
        return ModifiedDataset(dataset)

#
# ~~~ Add dropout to a standard ReLU network
def add_dropout_to_sequential_relu_network( add_dropout_to_sequential_relu_network, p=0.5 ):
    layers = []
    for layer in add_dropout_to_sequential_relu_network:
        layers.append(layer)
        if isinstance(layer, torch.nn.ReLU):
            layers.append(torch.nn.Dropout(p=p))
    return torch.nn.Sequential(*layers)

#
# ~~~ Generate a list of batch sizes
def get_batch_sizes( N, b ):
    quotient = N // b
    remainder = N % b
    extra = [remainder] if remainder>0 else []
    batch_sizes = [b]*quotient + extra
    assert sum(batch_sizes)==N
    return batch_sizes

def k_smallest_indices( dataframe, column, k ):
    data = dataframe if isinstance(dataframe,pd.Series) else dataframe[column].array
    return np.argpartition(data, k)[:k]

def k_largest_indices( dataframe, column, k ):
    data = dataframe if isinstance(dataframe,pd.Series) else dataframe[column].array
    return np.argpartition(-data, k)[:k]

#
# ~~~ Load a the predictions trained model, based on the dataframe of results you get from hyperparameter search
def get_predictions_and_targets( dataframe, i ):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    x_val   = data.x_val.to( device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE )
    targets = data.y_val.to( device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE )
    bnn = load_trained_model_from_dataframe(dataframe,i)
    with torch.no_grad(): predictions = bnn( x_val, n=data.iloc[i].N_POSTERIOR_SAMPLES )
    return predictions, targets

#
# ~~~ Try to get dict[key] but, if that doesn't work, then get source_of_default.default_key instead.
def get_key_or_default( dictionary, key, default ):
    try: return dictionary[key]
    except KeyError:
        my_warn(f'Hyper-parameter "{key}" not specified. Using default value of {default}.')
        return default

#
# ~~~ Use argparse to extract the file name `my_hyperparmeters.json` and such from `python train_<algorithm>.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
def parse(hint=None):
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument( '--json', type=str, required=True )
        parser.add_argument( '--model_save_dir', type=str )
        parser.add_argument( '--final_test', action=argparse.BooleanOptionalAction )
        parser.add_argument( '--overwrite_json', action=argparse.BooleanOptionalAction )
        args = parser.parse_args()
    except:
        if hint is not None: print(f"\n\n    Hint: {hint}\n")
        raise
    input_json_filename = args.json if args.json.endswith(".json") else args.json+".json"
    model_save_dir      = args.model_save_dir
    final_test          = (args.final_test is not None)
    overwrite_json      = (args.overwrite_json is not None)
    return input_json_filename, model_save_dir, final_test, overwrite_json
