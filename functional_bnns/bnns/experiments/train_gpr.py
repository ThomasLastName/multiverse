### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from matplotlib import pyplot as plt
from importlib import import_module
import argparse

#
# ~~~ Other parts of this package
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils.handling import (
    generate_json_filename,
    dict_to_json,
    json_to_dict,
    my_warn,
    convert_Dataset_to_Tensors,
)
from bnns.utils.plotting import plot_gpr


### ~~~
## ~~~ Config/setup
### ~~~

#
# ~~~ Template for what the dictionary of hyperparmeters should look like
hyperparameter_template = {
    #
    # ~~~ Misc.
    "DEVICE": "cpu",
    "dtype": "float",
    "seed": 2024,
    #
    # ~~~ Which problem
    "data": "univar_missing_middle",
    #
    # ~~~ For training
    "conditional_std": 0.19,
    "bw": 0.1,
    #
    # ~~~ For metrics and visualization
    "extra_std": True,
    "show_diagnostics": True,
}

#
# ~~~ Use argparse to extract the file name from `python gpr.py --json my_hyperparmeters.json` (https://stackoverflow.com/a/67731094)
parser = argparse.ArgumentParser()
try:
    parser.add_argument("--json", type=str, required=True)
    input_json_filename = parser.parse_args().json
    input_json_filename = (
        input_json_filename
        if input_json_filename.endswith(".json")
        else input_json_filename + ".json"
    )
except:
    print("")
    print("    Hint: try `python gpr.py --json demo_gpr`")
    print("")
    raise

#
# ~~~ Load the .json file into a dictionary
hyperparameters = json_to_dict(input_json_filename)

#
# ~~~ Load the dictionary's key/value pairs into the global namespace
globals().update(
    hyperparameters
)  # ~~~ e.g., if hyperparameters=={ "a":1, "B":2 }, then this defines a=1 and B=2

#
# ~~~ Might as well fix a seed
torch.manual_seed(seed)

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you can't save it as .json)
dtype = getattr(torch, dtype)  # ~~~ e.g., "float" (str) -> torch.float (torch.dtype)
torch.set_default_dtype(dtype)

#
# ~~~ Load the data
try:
    data = import_module(
        f"bnns.data.{data}"
    )  # ~~~ this is equivalent to `import bnns.data.<data> as data`
except:
    data = import_module(data)

x_train, y_train = convert_Dataset_to_Tensors(
    data.D_train
)  # ~~~ for generality, convert D_train to tensors instead of attempting to import x_train and y_train directly
x_test, _ = convert_Dataset_to_Tensors(
    data.D_test
)  # ~~~ for generality, convert D_train to tensors instead of attempting to import x_test directly
x_train = x_train.to(device=DEVICE, dtype=dtype)
y_train = y_train.to(device=DEVICE, dtype=dtype)
x_test = x_test.to(device=DEVICE, dtype=dtype)
data_is_univariate = data.D_train[0][0].numel() == 1


### ~~~
## ~~~ Do GPR
### ~~~

#
# ~~~ Borrow from SSGE, the implementation of the sub-routines responsible for building the kernel matrix and estimating a good kernel bandwidth
kernel_matrix = SSGE_backend().gram_matrix
bandwidth_estimator = SSGE_backend().heuristic_sigma

#
# ~~~ Do GPR
K_in = scale * kernel_matrix(x_train.unsqueeze(-1), x_train.unsqueeze(-1), bw)
K_out = scale * kernel_matrix(x_test.unsqueeze(-1), x_test.unsqueeze(-1), bw)
K_btwn = scale * kernel_matrix(x_test.unsqueeze(-1), x_train.unsqueeze(-1), bw)
with torch.no_grad():
    sigma2 = torch.tensor(conditional_std) ** 2

K_inv = torch.linalg.inv(K_in + sigma2 * torch.eye(len(x_train), device=DEVICE))
posterior_mean = (K_btwn @ K_inv @ y_train).squeeze()
posterior_std = (K_out - K_btwn @ K_inv @ K_btwn.T).diag().sqrt()

#
# ~~~ Some plotting stuff
if data_is_univariate:
    grid = data.x_test.to(device=DEVICE, dtype=dtype)
    green_curve = data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig, ax = plot_gpr(
        fig,
        ax,
        grid,
        green_curve,
        x_train_cpu,
        y_train_cpu,
        mean=(posterior_mean + sigma2 if extra_std else posterior_mean),
        std=posterior_std,
        predictions_include_conditional_std=extra_std,
    )
    plt.show()


### ~~~
## ~~~ Evaluate the trained model
### ~~~

hyperparameters["metric"] = "here, we will record metrics"


### ~~~
## ~~~ Save the results
### ~~~

if input_json_filename.startswith("demo"):
    my_warn(
        f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)'
    )
else:
    output_json_filename = generate_json_filename()
    dict_to_json(hyperparameters, output_json_filename)

#
