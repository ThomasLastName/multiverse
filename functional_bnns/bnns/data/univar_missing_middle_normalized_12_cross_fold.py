from bnns.data.univar_missing_middle_cross_fold import (
    x_train,
    y_train,
    x_test,
    y_test,
    x_val,
    y_val,
)
from bnns.data.univar_missing_middle import (
    f,
    grid,
    interpolary_grid,
    extrapolary_grid,
    random_points_only,
    data_only,
    current_batch_and_random_data,
)
from bnns.utils.handling import convert_Tensors_to_Dataset

#
# ~~~ Scale down the data
scale = 12
y_train /= scale
y_test /= scale
y_val /= scale
ground_truth = lambda x: f(x) / scale

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train, y_train)
D_test = convert_Tensors_to_Dataset(x_test, y_test)
D_val = convert_Tensors_to_Dataset(x_val, y_val)
