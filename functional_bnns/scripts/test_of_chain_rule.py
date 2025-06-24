from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from quality_of_life.my_base_utils import support_for_progress_bars

import torch
from torch import nn
from bnns import GaussianBNN
from bnns.data.univar_missing_middle import x_train, y_train, x_test
from bnns.utils import compute_Jacobian_of_flattened_model

#
# ~~~ Process Data
device = "cuda" if torch.cuda.is_available() else "cpu"
x_train = x_train.reshape(-1, 1).to(device)
y_train = y_train.to(device)
x_test = x_test.reshape(-1, 1).to(device)


#
# ~~~ Do model.parameters().grad+=flat_grad
def accumulate_grad(model, flat_grad):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad += flat_grad[offset : offset + numel].view_as(p)
        offset += numel


#
# ~~~ Define measurement set sampler
class BNN(GaussianBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resample_measurement_set(self, n=64):
        device, dtype = self.infer_device_and_dtype()
        self.measurement_set = (
            torch.rand(size=(n, 1), device=device, dtype=dtype) * 2 - 1
        )


#
# ~~~ Training
n_epochs = 200
for Jac in (True, False):
    #
    # ~~~ Instantiate model
    torch.manual_seed(2025)
    model = nn.Sequential(
        nn.Linear(1, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1),
    )
    bnn = BNN(*model).to(device)
    optimizer = torch.optim.Adam(bnn.parameters(), lr=0.0001)
    desc = (
        "fBNN Computing the Full Jacobian"
        if Jac
        else "fBNN Without Computing the Jacobian Explicitly"
    )
    with support_for_progress_bars():
        pbar = tqdm(total=n_epochs, desc=desc)
        tick = time()
        for iter in range(n_epochs):
            log_lik = bnn.estimate_expected_log_likelihood(x_train, y_train)
            if not Jac:
                kl_div = bnn.functional_kl()
                loss = kl_div - log_lik
                loss.backward()
                optimizer.step()
            if Jac:
                with torch.no_grad():
                    _, posterior_score, prior_score = bnn.functional_kl(
                        return_raw_ingredients=True
                    )
                    g = posterior_score - prior_score
                    J = compute_Jacobian_of_flattened_model(
                        bnn, bnn.measurement_set, out_features=1
                    )
                    flattened_kl_grad = (g @ J).squeeze()
                loss = -log_lik
                loss.backward()
                accumulate_grad(bnn, flattened_kl_grad)
                optimizer.step()
            bnn.resample_weights()
            pbar.set_postfix({"loss": f"{loss.item():4.4}"})
            pbar.update()
        pbar.close()
        tock = time()
        with torch.no_grad():
            if Jac:
                predictions_Jac = bnn(x_test, n=100)
                time_Jac = tock - tick
            else:
                predictions_trick = bnn(x_test, n=100)
                time_trick = tock - tick

abs_error = (predictions_Jac - predictions_trick).abs().mean()
rel_error = abs_error / predictions_Jac.abs().mean()
print(f"Results are effectively the same with relative error {rel_error.item()}.")
print(
    f"The computation time of our poposed method is {1+time_trick/time_Jac:.2f} time as fast."
)
