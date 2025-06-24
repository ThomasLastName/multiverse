import torch
from bnns.utils.handling import my_warn, convert_Tensors_to_Dataset

#
# ~~~ Make up some fake data
torch.manual_seed(2024)
f = lambda x: x * torch.sin(2 * torch.pi * x)
x_train = torch.linspace(-1, 1, 41)
y_train = (f(x_train) + 0.2 * torch.randn_like(x_train)).reshape(-1, 1)
x_val = torch.linspace(-1, 1, 41)[1:] - 1 / 40
y_val = (f(x_val) + 0.2 * torch.randn_like(x_val)).reshape(-1, 1)
x_test = torch.linspace(-1, 1, 301)
y_test = (f(x_test)).reshape(-1, 1)
grid = x_test
ground_truth = f

#
# ~~~ Package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train, y_train)
D_test = convert_Tensors_to_Dataset(x_test, y_test)
D_val = convert_Tensors_to_Dataset(x_val, y_val)


### ~~~
## ~~~ Define procedures by which to sample measurement sets
### ~~~

lo = grid.min()
hi = grid.max()


#
# ~~~ Generate completely random points to be the measurement set
def random_points_only(self, n):
    device, dtype = self.infer_device_and_dtype()
    self.measurement_set = torch.rand(n, device=device, dtype=dtype) * (hi - lo) + lo


#
# ~~~ Just take the entire training dataset to be the measurement set
def data_only(self, n=None):
    if not hasattr(self, "measurement_set"):
        device, dtype = self.infer_device_and_dtype()
        self.measurement_set = x_train.to(device=device, dtype=dtype)


#
# ~~~ Use the current batch of training data along with
def current_batch_and_random_data(self, n):
    device, dtype = self.infer_device_and_dtype()
    if hasattr(self, "desired_measurement_points"):
        batch_size = len(self.desired_measurement_points)
        if batch_size > n:
            my_warn(
                "More desired measurement points are specified than the total number of measurement points (this is most likely the training result batch size exceeding the specified number of measurement points). Only a randomly chosen subset of the desired measurement points will be used."
            )
            self.measurement_set = self.desired_measurement_points[
                torch.randperm(batch_size)[:n]
            ]
        else:
            self.measurement_set = torch.concatenate(
                [
                    self.desired_measurement_points,
                    torch.rand(n - batch_size, device=device, dtype=dtype) * (hi - lo)
                    + lo,
                ]
            )
            if n - batch_size <= 10 and not hasattr(
                self, "already_warned_that_n_meas_too_small"
            ):
                my_warn(
                    "There are almost as many `desired_measurement_points` as total measurement points. Please consider using slightly more measurement points."
                )
                self.already_warned_that_n_meas_too_small = True
    else:
        my_warn(
            "Failed to find training data batch to be included in the measurement set. Please verify that `use_input_in_next_measurement_set=True` in `estimate_expected_log_likelihood(X,y,use_input_in_next_measurement_set)` and that this is called before the kl is computed."
        )
        self.measurement_set = (
            torch.rand(n, device=device, dtype=dtype) * (hi - lo) + lo
        )
