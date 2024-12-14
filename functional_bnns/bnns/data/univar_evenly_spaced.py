
import torch
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset

#
# ~~~ Make up some fake data
torch.manual_seed(2024)
f = lambda x: x * torch.sin(2*torch.pi*x)
x_train = torch.linspace(-1,1,41)
y_train = (f(x_train) + 0.2*torch.randn_like(x_train)).reshape(-1,1)
x_val   = torch.linspace(-1,1,41)[1:]- 1/40
y_val   = (f(x_val) + 0.2*torch.randn_like(x_val)).reshape(-1,1)
x_test  = torch.linspace(-1,1,301)
y_test  = (f(x_test)).reshape(-1,1)

#
# ~~~ Package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
