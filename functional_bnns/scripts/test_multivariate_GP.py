
import torch
from bnns.models.bivar_BNN_GP_prior import BNN
from bnns.data.bivar_trivial import x_test
from matplotlib import pyplot as plt

BNN.GP.etas *= 0.
BNN.GP.bandwidths = torch.tensor(BNN.out_features*[0.1])

try:
    y_prior = BNN.prior_forward(x_test)
except torch._C._LinAlgError:
    try:
        BNN.GP.etas += 0.00001
        y_prior = BNN.prior_forward(x_test)
        print("")
        print(f"eta={BNN.GP.etas[0]} achieved numerical stability with torch.float")
        print("")
    except torch._C._LinAlgError:
        BNN = BNN.to(torch.double)
        x_test = x_test.to(torch.double)
        y_prior = BNN.prior_forward(x_test)
        print("")
        print(f"eta={BNN.GP.etas[0]} achieved numerical stability with torch.double, but not with torch.float")
        print("")


x =  x_test[:,0].squeeze().cpu().numpy()
y = y_prior[:,0].squeeze().cpu().numpy()
plt.plot(x,y)
plt.show()

