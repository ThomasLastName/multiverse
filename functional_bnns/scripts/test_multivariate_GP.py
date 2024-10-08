
import torch
from bnns.models.bivar_BNN_GP_prior import BNN
from bnns.data.bivar_trivial import x_test
from matplotlib import pyplot as plt
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
kernel_matrix = SSGE_backend().gram_matrix


BNN.prior_GP_kernel_eta *= 0.

try:
    y_prior = BNN.prior_forward(x_test)
except torch._C._LinAlgError:
    try:
        BNN.prior_GP_kernel_eta += 0.00001
        y_prior = BNN.prior_forward(x_test)
    except torch._C._LinAlgError:
        print("Simply adding noise did not suffice")
        BNN = BNN.to(torch.double)
        x_test = x_test.to(torch.double)
        y_prior = BNN.prior_forward(x_test)


x =  x_test[:,0].squeeze().cpu().numpy()
y = y_prior[:,0].squeeze().cpu().numpy()
plt.plot(x,y)
plt.show()

