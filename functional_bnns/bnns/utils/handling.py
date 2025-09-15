import numpy as np
import pandas as pd
import torch
from torch import nn
import platform
import sys
import os
import re
import json
import pytz
import argparse
from tqdm import tqdm
from glob import glob
from datetime import datetime
from importlib import import_module


### ~~~
## ~~~ Non-math non-plotting stuff (e.g., data processing)
### ~~~

#
# ~~~ Flatten and concatenate all the parameters in a model
flatten_parameters = lambda model: torch.cat([p.view(-1) for p in model.parameters()])


#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def convert_to_list_and_check_items(x, classes, other_requirement=lambda *args: True):
    #
    # ~~~ Convert x to a list (if x is already a list, this has no effect)
    try:
        X = list(x)
    except TypeError:
        X = [x]
    except:
        raise
    assert isinstance(
        X, list
    ), "Unexpected error: both list(x) and [x] failed to create a list out of x."
    #
    # ~~~ Verify the type of each item in the list
    for item in X:
        assert isinstance(item, classes)
        assert other_requirement(item), "The user supplied check was not satisfied."
    #
    # ~~~ Return the list whose items all meet the type and other requirements
    return X


#
# ~~~ Convert x to [x] if x isn't a list to begin with, then verify the type of each item of x, along with any other user-specified requirement
def non_negative_list(x, integer_only=False):
    return convert_to_list_and_check_items(
        x=x,
        classes=int if integer_only else (int, float),
        other_requirement=lambda item: item >= 0,
    )


#
# ~~~ A standard early stopping rule (https://stackoverflow.com/a/73704579/11595884)
class EarlyStopper:
    def __init__(self, patience=20, delta=0.05):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_count = 0
        self.min_val_loss = float("inf")

    def __call__(self, val_loss):
        #
        # ~~~ If the validation loss is decreasing, reset the counter
        if val_loss <= self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        #
        # ~~~ Compute the relative difference between the current `val_loss` and smallest yet `self.min_val_loss`
        if self.min_val_loss > 0:
            rel_val_loss = (
                val_loss / self.min_val_loss - 1
            )  # ~~~ `== ( val_loss - self.min_val_loss ) / self.min_val_loss` but more numerically stable
        else:
            rel_val_loss = abs(val_loss - self.min_val_loss) / abs(self.min_val_loss)
        #
        # ~~~ If the current val_loss is "more than delta worse" relative to min_val_loss, increment the counter
        if rel_val_loss > self.delta:
            self.counter += 1
            self.max_count = max(self.counter, self.max_count)
        #
        # ~~~ Stop iff max_count >= patience
        if self.max_count >= self.patience:
            return True
        else:
            return False


#
# ~~~ Load all the .json files in a directory to data frame
def load_filtered_json_files(directory, verbose=True):
    #
    # ~~~ Load (as a list of dictionaries) all the .json files in a directory that don't start with "RUN_THIS"
    with support_for_progress_bars():
        json_files = glob(os.path.join(directory, "*.json"))
        json_files = [
            json
            for json in json_files
            if not get_file_name(json).startswith("RUN_THIS")
        ]
        all_dicts = [
            json_to_dict(json)
            for json in (
                tqdm(json_files, desc="Loading json files") if verbose else json_files
            )
        ]
    #
    # ~~~ Remove from each dictionary any key/value pair where the value is a list, as pandas doesn't like those
    all_filtered_dicts = [
        {k: v for k, v in dict.items() if not isinstance(v, list)} for dict in all_dicts
    ]
    return pd.DataFrame(all_filtered_dicts)


#
# ~~~ Infer the width of each model
def infer_width_and_depth(dataframe, field="ARCHITECTURE"):
    #
    # ~~~ Infer the width of each model
    width_mapping = {}
    for model in dataframe[field].unique():
        text_after_last_underscore = model[
            model.rfind("_") + 1 :
        ]  # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        width_mapping[model] = int(text_after_last_underscore)
    dataframe["width"] = dataframe[field].map(width_mapping)
    #
    # ~~~ Infer the depth of each model
    depth_mapping = {}
    for model in dataframe[field].unique():
        how_many_underscores = (
            len(model.split("_")) - 1
        )  # ~~~ e.g., if model=="univar_NN.univar_NN_30_30_30", then text_after_last_underscore=="30"
        depth_mapping[model] = (
            how_many_underscores - 2 if how_many_underscores > 1 else 2
        )
    dataframe["depth"] = dataframe[field].map(depth_mapping)
    return dataframe


#
# ~~~ Get the dataframe.iloc[i,"arg"] for all the arguments `args`
def get_attributes_from_row_i(dataframe, i, *args):
    return [dataframe.iloc[i][arg] for arg in args]


#
# ~~~ Filter a dataframe by multiple attributes
def filter_by_attributes(dataframe, **kwargs):
    filtered_results = dataframe
    for key, value in kwargs.items():
        filtered_results = filtered_results[filtered_results[key] == value]
    return filtered_results


#
# ~~~ Load a trained BNN, based on the string `architecture` that points to the file where the model is defined
def load_trained_bnn(architecture: str, model: str, state_dict_path):
    #
    # ~~~ Load the untrained model
    import bnns

    architecture = import_module(
        f"bnns.models.{architecture}"
    ).NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    model = getattr(bnns, model)(*architecture)
    model.load_state_dict(torch.load(state_dict_path))
    return model


#
# ~~~ Load a trained ensemble, based on the string `architecture` that points to the file where the model is defined
def load_trained_ensemble(architecture: str, n_models: int, state_dict_path):
    #
    # ~~~ Load the untrained model
    import bnns

    architecture = import_module(
        f"bnns.models.{architecture}"
    ).NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    ensemble = bnns.Ensemble.SequentialSteinEnsemble(architecture, n_models)
    ensemble.load_state_dict(torch.load(state_dict_path))
    ensemble.parameters_have_been_updated = True
    return ensemble


#
# ~~~ Load a trained conventional neural network, based on the dataframe of results you get from hyperparameter search
def load_trained_nn(architecture: str, state_dict_path):
    import bnns

    architecture = import_module(
        f"bnns.models.{architecture}"
    ).NN  # ~~~ e.g., architecture=="my_model" points to a file `my_model.py` in the `models` folder
    architecture.load_state_dict(torch.load(state_dict_path))
    return architecture


#
# ~~~ Load a trained model, based on the dataframe of results you get from hyperparameter search
def load_trained_model_from_dataframe(results_dataframe, i):
    #
    # ~~~ Load the untrained model
    architecture = results_dataframe.iloc[i].ARCHITECTURE
    state_dict_path = results_dataframe.iloc[i].STATE_DICT_PATH
    try:
        model = results_dataframe.iloc[i].MODEL
        return load_trained_bnn(architecture, model, state_dict_path)
    except:
        try:
            n_models = results_dataframe.iloc[i].N_MODELS
            return load_trained_ensemble(architecture, n_models, state_dict_path)
        except:
            return load_trained_nn(architecture, state_dict_path)


#
# ~~~ Generate a .json filename based on the current datetime
def generate_json_filename(verbose=True, message=None):
    #
    # ~~~ Generate a .json filename
    time = datetime.now(pytz.timezone("US/Central"))  # ~~~ current date and time CST
    file_name = str(time)
    file_name = file_name[
        : file_name.find(".")
    ]  # ~~~ remove the number of milliseconds (indicated with ".")
    file_name = file_name.replace(" ", "_").replace(
        ":", "-"
    )  # ~~~ replace blank space (between date and time) with an underscore and colons (hr:mm:ss) with dashes
    file_name = process_for_saving(
        file_name + ".json"
    )  # ~~~ procsess_for_saving("path_that_exists.json") returns "path_that_exists (1).json"
    #
    # ~~~ Craft a message to print
    if verbose:
        if time.hour > 12:
            hour = time.hour - 12
            suffix = "pm"
        else:
            hour = time.hour
            suffix = "am"
        base_message = (
            bcolors.OKBLUE
            + f"    Generating file name {file_name} at {hour}:{time.minute:02d}{suffix} CST"
            + bcolors.HEADER
        )
        if message is not None:
            if not message[0] == " ":
                message = " " + message
            base_message += message
        print(base_message)
    return file_name


#
# ~~~ My version of the missing feature: a `dataset.to` method
def set_Dataset_attributes(dataset, device, dtype):
    try:
        #
        # ~~~ Directly access and modify the underlying tensors
        dataset.X = dataset.X.to(device=device, dtype=dtype)
        dataset.y = dataset.y.to(device=device, dtype=dtype)
        return dataset
    except AttributeError:
        #
        # ~~~ Redefine the __getattr__ method (this is hacky; I don't know a better way; also, chat-gpt proposed this)
        class ModifiedDataset(torch.utils.data.Dataset):
            def __init__(self, original_dataset):
                self.original_dataset = original_dataset
                self.device = device
                self.dtype = dtype

            def __getitem__(self, index):
                x, y = self.original_dataset[index]
                return x.to(device=self.device, dtype=self.dtype), y.to(
                    device=self.device, dtype=self.dtype
                )

            def __len__(self):
                return len(self.original_dataset)

        return ModifiedDataset(dataset)


#
# ~~~ Add dropout to a standard ReLU network
def add_dropout_to_sequential_relu_network(sequential_relu_network, p=0.5):
    layers = []
    for layer in sequential_relu_network:
        layers.append(layer)
        if isinstance(layer, nn.ReLU):
            layers.append(nn.Dropout(p=p))
    return nn.Sequential(*layers)


#
# ~~~ Generate a list of batch sizes
def get_batch_sizes(N, b):
    quotient = N // b
    remainder = N % b
    extra = [remainder] if remainder > 0 else []
    batch_sizes = [b] * quotient + extra
    assert sum(batch_sizes) == N
    return batch_sizes


def k_smallest_indices(dataframe, column, k):
    data = dataframe if isinstance(dataframe, pd.Series) else dataframe[column].array
    return np.argpartition(data, k)[:k]


def k_largest_indices(dataframe, column, k):
    data = dataframe if isinstance(dataframe, pd.Series) else dataframe[column].array
    return np.argpartition(-data, k)[:k]


#
# ~~~ Load a the predictions trained model, based on the dataframe of results you get from hyperparameter search
def get_predictions_and_targets(dataframe, i):
    data = import_module(f"bnns.data.{dataframe.iloc[i].DATA}")
    x_val = data.x_val.to(device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE)
    targets = data.y_val.to(device=data.iloc[i].DEVICE, dtype=data.iloc[i].DTYPE)
    bnn = load_trained_model_from_dataframe(dataframe, i)
    with torch.no_grad():
        predictions = bnn(x_val, n=data.iloc[i].N_POSTERIOR_SAMPLES)
    return predictions, targets


#
# ~~~ Try to get dict[key] but, if that doesn't work, then get source_of_default.default_key instead.
def get_key_or_default(dictionary, key, default):
    try:
        return dictionary[key]
    except KeyError:
        my_warn(
            f'Hyper-parameter "{key}" not specified. Using default value of {default}.'
        )
        return default


#
# ~~~ Use argparse to extract the file name `my_hyperparmeters.json` and such from `python train_<algorithm>.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
def parse(hint=None):
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument("--json", type=str, required=True)
        parser.add_argument("--model_save_dir", type=str)
        parser.add_argument("--final_test", action=argparse.BooleanOptionalAction)
        parser.add_argument("--overwrite_json", action=argparse.BooleanOptionalAction)
        args = parser.parse_args()
    except:
        if hint is not None:
            print(f"\n\n    Hint: {hint}\n")
        raise
    input_json_filename = (
        args.json if args.json.endswith(".json") else args.json + ".json"
    )
    model_save_dir = args.model_save_dir
    final_test = args.final_test is not None
    overwrite_json = args.overwrite_json is not None
    return input_json_filename, model_save_dir, final_test, overwrite_json


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/my_base_utils.py
### ~~~


#
# ~~~ Format a long list for printing
def format_value(value):
    if isinstance(value, list) and len(value) > 4:
        #
        # ~~~ Show only the first two and last two elements
        return [value[0], value[1], "...", value[-2], value[-1]]
    return value


#
# ~~~ Pretty print a dictionary; from https://www.geeksforgeeks.org/python-pretty-print-a-dictionary-with-dictionary-value/
fdict = lambda dict: json.dumps({k: format_value(v) for k, v in dict.items()}, indent=4)
print_dict = lambda dict: print(fdict(dict))


#
# ~~~ Save a dictionary as a .json; from https://stackoverflow.com/a/7100202/11595884
def dict_to_json(dict, path_including_file_extension, override=False, verbose=True):
    #
    # ~~~ Check that the path is available
    not_empty = os.path.exists(path_including_file_extension)
    #
    # ~~~ If that path already exists and the user did not say "over-ride" it, then raise an error
    if not_empty and not override:
        raise ValueError(
            "The specified path already exists. Operation halted. Specify `override=True` to override this halting."
        )
    #
    # ~~~ If the file path is either available, or the user gave permission to over-ride it, then proceed to write there
    with open(path_including_file_extension, "w") as fp:
        json.dump(dict, fp, indent=4)
    #
    # ~~~ Print helpful messages
    if verbose:
        if override:
            my_warn(
                f"The path {path_including_file_extension} was not empty. It has been overwritten."
            )
        print(
            f"    Created {path_including_file_extension} at {os.path.abspath(path_including_file_extension)}:\n"
        )


#
# ~~~ Load a .json as a dictionary (https://chatgpt.com/share/683b4261-cef8-8001-8ca5-d63a2cb637b2)
def json_to_dict(path_including_file_extension):
    with open(path_including_file_extension, "r") as fp:
        #
        # ~~~ Remove // comments from the end of the line (or whole line if it starts with // after leading spaces)
        content = "".join(
            re.sub(r"//.*", "", line)
            for line in fp
            if not line.strip().startswith("//")
        )
    return json.loads(content)


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]


def get_file_name(file_path):
    return os.path.basename(file_path)


#
# ~~~ Turn "name of file that already eists.txt" into "name of file that already exists (1).txt"
def modify_path(file_path_or_name, force=False):
    #
    # ~~~ If the path doesn't exist, then no modification is needed; do nothing
    if (not os.path.exists(file_path_or_name)) and (not force):
        return file_path_or_name
    #
    # ~~~ Remove any the doodads surring the file name, thus leaving the only thing we wish to modify
    original_extension = get_file_extension(file_path_or_name)
    file_name_and_extnesion = get_file_name(file_path_or_name)
    name_only = file_name_and_extnesion.replace(original_extension, "")
    #
    # ~~~ Check if the name ends with " (anything)"
    start = name_only.rfind("(")
    end = name_only.rfind(")")
    correct_format = name_only.endswith(")") and (
        not start == -1
    )  # and name_only[start-1]==" "   # ~~~ note: .rfind( "(" ) returns -1 if "(" is not found
    #
    # ~~~ If the file name is like "text (2)", turn that into "text (3)"
    if correct_format:
        if correct_format:
            thing_inside_the_parentheses = name_only[start + 1 : end]
            try:
                num = int(thing_inside_the_parentheses)
                new_num = num + 1
                modified_name = name_only[: start + 1] + str(new_num) + name_only[end:]
            except ValueError:
                #
                # ~~~ If conversion to int fails, treat it as if the name didn't end with a valid " (n)"
                correct_format = False
    #
    # ~~~ If the file name didn't end with " (n)" for some n, then just append " (1)" to the file name
    if not correct_format:
        modified_name = name_only + " (1)"
    #
    # ~~~ Reattach any doodads we removed
    return file_path_or_name.replace(
        file_name_and_extnesion, modified_name + original_extension
    )


#
# ~~~ Turn "name of file that already eists.txt" into "name of file that already exists (1).txt", and also "name of file that already exists (1).txt" into "name of file that already exists (2).txt", etc.
def process_for_saving(file_path_or_name):
    while os.path.exists(file_path_or_name):
        file_path_or_name = modify_path(file_path_or_name)
    return file_path_or_name


#
# ~~~ Optionally, load functions that further manipulate the color of console output
try:
    from quality_of_life.my_base_utils import support_for_progress_bars, my_warn
except:  # ~~~ however, if those functions are not available, then let their definitions be trivial (for compatibility)
    import warnings

    my_warn = warnings.warn
    from contextlib import contextmanager

    @contextmanager
    def support_for_progress_bars():
        yield


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/ansi.py
### ~~~

#
# ~~~ Set up colored printing in the console
if (
    platform.system() == "Windows"
):  # platform.system() returns the OS python is running o0n | see https://stackoverflow.com/q/1854/11595884
    os.system(
        "color"
    )  # Makes ANSI codes work | see Szabolcs' comment on https://stackoverflow.com/a/15170325/11595884


class bcolors:  # https://stackoverflow.com/a/287944/11595884
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/my_torch_utils.py
### ~~~


#
# ~~~ Extract the raw tensors from a pytorch Dataset
def convert_Dataset_to_Tensors(object_of_class_Dataset, batch_size=None):
    assert isinstance(object_of_class_Dataset, torch.utils.data.Dataset)
    if isinstance(object_of_class_Dataset, convert_Tensors_to_Dataset):
        return object_of_class_Dataset.X, object_of_class_Dataset.y
    else:
        n_data = len(object_of_class_Dataset)
        b = n_data if batch_size is None else batch_size
        return next(
            iter(torch.utils.data.DataLoader(object_of_class_Dataset, batch_size=b))
        )  # return the actual tuple (X,y)


#
# ~~~ Convert Tensors into a pytorch Dataset; from https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
class convert_Tensors_to_Dataset(torch.utils.data.Dataset):
    #
    # ~~~ Define attributes
    def __init__(
        self,
        X_tensor,
        y_tensor,
        X_transforms_list=None,
        y_transforms_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert X_tensor.shape[0] == y_tensor.shape[0]
        self.X = X_tensor
        self.y = y_tensor
        self.X_transforms = X_transforms_list
        self.y_transforms = y_transforms_list

    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have to enable indexing; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __getitem__(self, index):
        x = self.X[index]
        if self.X_transforms is not None:
            for transform in self.X_transforms:
                x = transform(x)
        y = self.y[index]
        if self.y_transforms is not None:
            for transform in self.y_transforms:
                y = transform(y)
        return x, y

    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __len__(self):
        return self.y.shape[0]


#
# ~~~ Get all available gradients of the parameters in a pytorch model
def get_flat_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


#
# ~~~ Given a flat vector of desired gradients, and a model, assign those to the .grad attribute of the model's parameters
def set_flat_grads(model, flat_grads):
    # TODO: a safety feature checking the shape/class of flat_grads (should be a 1d Torch vector)
    start = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.grad.numel()
            p.grad.data = flat_grads[start : start + numel].view_as(p.grad)
            start += numel
    if start > len(flat_grads):
        my_warn(
            f"The lenght of the supplied vector [{len(flat_grads)}] exceeds the number of parameters in the model which require grad [{start}]"
        )


#
# ~~~ Helper function which creates a new instance of the supplied sequential architeture
def nonredundant_copy_of_module_list(module_list, sequential=False):
    architecture = [(type(layer), layer) for layer in module_list]
    layers = []
    for layer_type, layer in architecture:
        if layer_type == nn.Linear:
            #
            # ~~~ For linear layers, create a brand new linear layer of the same size independent of the original
            layers.append(
                nn.Linear(
                    layer.in_features, layer.out_features, bias=(layer.bias is not None)
                )
            )
        else:
            #
            # ~~~ For other layers (activations, Flatten, softmax, etc.) just copy it
            layers.append(layer)
    return nn.Sequential(*layers) if sequential else nn.ModuleList(layers)


#
# ~~~ Discern whether or not the provided layer has exactly two parameters: a weight parameter, and a bias parameter (e.g., a Conv2d instance should also pass)
def is_weight_and_bias_layer(layer):
    params = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())
    has_weight = 'weight' in params # ~~~ must have a weight parameter, e.g., to rule out ReLU layers
    only_weight_and_bias = set(params.keys()).issubset({'weight', 'bias'}) # ~~~ only allow weight and optional bias, and nothing else
    no_buffers = len(buffers) == 0
    return has_weight and only_weight_and_bias and no_buffers

