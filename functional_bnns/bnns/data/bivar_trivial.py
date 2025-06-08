
import torch
torch.manual_seed(1234)
from bnns.utils.handling import convert_Tensors_to_Dataset
from bnns.data.univar_missing_middle import x_train, x_test, x_val, y_train, y_test, y_val, f

#
# ~~~ Data settings
in_dim = 2
out_dim = 2

x_train = torch.column_stack(in_dim*[x_train])
x_test = torch.column_stack(in_dim*[x_test])
x_val = torch.column_stack(in_dim*[x_val])

y_train = torch.column_stack(out_dim*[y_train])
y_test = torch.column_stack(out_dim*[y_test])
y_val = torch.column_stack(out_dim*[y_val])


#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)