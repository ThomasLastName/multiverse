import math
import numpy as np
from scipy.integrate import quad
import torch
from torch import nn
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    calculate_gain,
)  # ~~~ used to define the prior distribution on network weights
from torch.func import jacrev, functional_call

from bnns.utils.handling import my_warn


### ~~~
## ~~~ Math stuff
### ~~~


#
# ~~~ Propose a good "prior" standard deviation for a parameter group
def std_per_param(p):
    if len(p.shape) == 2:
        #
        # ~~~ For weight matrices, use the standard deviation of pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
        gain = calculate_gain("relu")
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    elif len(p.shape) == 1:
        #
        # ~~~ For bias vectors, just use variance==1/len(p) because `_calculate_fan_in_and_fan_out` throws a ValueError(""Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"")
        numb_pars = len(p)
        std = 1 / math.sqrt(numb_pars)
    return torch.tensor(std, device=p.device, dtype=p.dtype)


#
# ~~~ Propose good a "prior" standard deviation for weights and biases of a linear layer; mimics pytorch's default initialization, but using a normal instead of uniform distribution (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
def std_per_layer(linear_layer):
    assert isinstance(linear_layer, nn.Linear)
    bound = 1 / math.sqrt(
        linear_layer.weight.size(1)
    )  # ~~~ see the link above (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
    std = bound / math.sqrt(
        3
    )  # ~~~ our reference distribution `uniform_(-bound,bound)` from the deafult pytorch weight initialization has standard deviation bound/sqrt(3), the value of which we copy
    return std


#
# ~~~ Define a class that extends a simple density function into a location scale distribution
class LocationScaleLogDensity:
    #
    # ~~~ Store the standard log density and test that it is, indeed, standard
    def __init__(self, standard_log_density, check_moments=True):
        self.standard_log_density = standard_log_density
        if check_moments:
            try:
                self.check_mean_zero_unit_variance()
            except:
                my_warn(
                    f"Unable to verify mean zero and unit variance in the standard log density {standard_log_density}. To surpress this warning, pass `check_moments=False` in the `__init__` method."
                )

    #
    # ~~~ Test that the supposedly "standard" log density has mean zero and unit variance
    def check_mean_zero_unit_variance(self, tol=1e-5):
        mean, err_mean = quad(
            lambda z: z * np.exp(self.standard_log_density(z)), -np.inf, np.inf
        )
        var, err_var = quad(
            lambda z: z**2 * np.exp(self.standard_log_density(z)), -np.inf, np.inf
        )
        if abs(mean) > tol or abs(var - 1) > tol or err_mean > tol or err_var > tol:
            raise RuntimeError(
                f"The mean is {mean} and the variance is {var} (should be 0 and 1)"
            )

    #
    # ~~~ Evaluate the log density of mu + sigma*z where z is distributed according to self.standard_log_density
    def __call__(self, where, mu, sigma, multivar=True):
        #
        # ~~~ Verify that `where-mu` will work
        try:
            assert mu.shape == where.shape
        except:
            assert isinstance(mu, (float, int))
        #
        # ~~~ Verify that `(where-mu)/sigma` will work
        try:
            assert isinstance(sigma, (float, int))
            assert sigma > 0
            sigma = torch.tensor(sigma, device=where.device, dtype=where.dtype)
        except:
            assert (
                len(sigma.shape) == 0 or sigma.shape == mu.shape
            )  # ~~~ either scalar, or a matrix of the same shape is `mu` and `where`
            assert (
                sigma > 0
            ).all(), f"Minimum standard deviation {sigma.min()} is not positive."
        #
        # ~~~ Compute the formula
        marginal_log_probs = self.standard_log_density(
            (where - mu) / sigma
        ) - torch.log(sigma)
        return marginal_log_probs.sum() if multivar else marginal_log_probs


#
# ~~~ Compute the log pdf of a multivariate normal distribution with independent coordinates
log_gaussian_pdf = LocationScaleLogDensity(
    lambda z: -(z**2) / 2 - math.log(math.sqrt(2 * torch.pi))
)


#
# ~~~ Compute the (appropriately shaped) Jacobian of the final layer of a nerural net (I came up with the formula for the Jacobian, and chat-gpt came up with the generalized vectorized pytorch implementation)
def manual_Jacobian(inputs_to_the_final_layer, number_of_output_features, bias=False):
    V = inputs_to_the_final_layer
    batch_size, width_of_the_final_layer = V.shape
    total_number_of_predictions = batch_size * number_of_output_features
    I = torch.eye(number_of_output_features, dtype=V.dtype, device=V.device)
    tiled_I = I.repeat(batch_size, 1)
    tiled_V = V.repeat_interleave(number_of_output_features, dim=0)
    Jac_wrt_weights = (tiled_I.unsqueeze(-1) * tiled_V.unsqueeze(1)).view(
        total_number_of_predictions, -1
    )
    Jac_wrt_biases = torch.tile(
        torch.eye(number_of_output_features), (batch_size, 1)
    ).to(device=Jac_wrt_weights.device, dtype=Jac_wrt_weights.dtype)
    return (
        Jac_wrt_weights
        if not bias
        else torch.column_stack([Jac_wrt_weights, Jac_wrt_biases])
    )


#
# ~~~ Compute the slope and intercept in linear regression
def lm(y, x):
    try:
        var = (x**2).mean() - x.mean() ** 2
        slope = (x * y).mean() / var - x.mean() * y.mean() / var
        intercept = y.mean() - slope * x.mean()
        return slope.item(), intercept.item()
    except:
        var = np.mean(x**2) - np.mean(x) ** 2
        slope = np.mean(x * y) / var - np.mean(x) * np.mean(y) / var
        intercept = np.mean(y) - slope * np.mean(x)
        return slope, intercept


#
# ~~~ Compute the empirical correlation coefficient between two vectors
def cor(u, w):
    try:
        stdstd = ((u**2).mean() - u.mean() ** 2).sqrt() * (
            (w**2).mean() - w.mean() ** 2
        ).sqrt()
        return ((u * w).mean() / stdstd - u.mean() * w.mean() / stdstd).item()
    except:
        return np.corrcoef(u, w)[0, 1]


#
# ~~~ Compute an empirical 95% confidence interval
iqr = (
    lambda tensor, dim=None: tensor.quantile(
        q=torch.Tensor([0.25, 0.75]).to(tensor.device), dim=dim
    )
    .diff(dim=0)
    .squeeze(dim=0)
)


#
# ~~~ Do polynomial regression
def univar_poly_fit(x, y, degree=1):
    try:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
    except:
        pass
    coeffs = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeffs)
    R_squared = cor(poly(x), y) ** 2
    return poly, coeffs, R_squared


#
# ~~~ Start with a grid of points in the unit cube, and then transform it to the desired bounds, includeing some exaggeration of the bounds
def process_grid_of_unit_cube(
    grid_of_unit_cube, bounds, extrapolation_percent=0.05, split=True
):
    lo = bounds[:, 0].clone()
    hi = bounds[:, 1].clone()
    range = hi - lo
    hi += extrapolation_percent * range
    lo -= extrapolation_percent * range
    grid = lo + (hi - lo) * grid_of_unit_cube
    extrapolary_grid = grid[
        torch.where(
            torch.logical_or(
                torch.any(grid > bounds[:, 1], dim=1),
                torch.any(grid < bounds[:, 0], dim=1),
            )
        )
    ]
    interpolary_grid = grid[
        torch.where(
            torch.logical_and(
                torch.all(grid <= bounds[:, 1], dim=1),
                torch.all(grid >= bounds[:, 0], dim=1),
            )
        )
    ]
    return (extrapolary_grid, interpolary_grid) if split else grid


#
# ~~~ Apply the exact formula for KL( N(mu_0,diag(sigma_0**2)) || N(mu_1,diag(sigma_1**2)) ) (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)
def diagonal_gaussian_kl(mu_0, sigma_0, mu_1, sigma_1):
    assert (
        mu_0.shape == mu_1.shape == sigma_0.shape == sigma_1.shape
    ), "Shape assumptions violated."
    assert (
        sigma_0.abs().min() > 0 and sigma_1.abs().min() > 0
    ), "Variance must be positive."
    return (
        (1 / 2)
        * (
            (
                (sigma_0 / sigma_1) ** 2
            ).sum()  # ~~~ the diagonal case of "tr(Sigma_1^{-1}Sigma_0)"
            - mu_0.numel()  # ~~~ what wikipedia calls "k"
            + (
                ((mu_1 - mu_0) / sigma_1) ** 2
            ).sum()  # ~~~ the diagonal case of "(mu_0-mu_1)^TSigma_1^{-1}(mu_0-mu_1)" potentially numerically unstble if mu_0\approx\mu_1 and \sigma_1 is small
        )
        + sigma_1.log().sum()
        - sigma_0.log().sum()
    )  # ~~~ the diagonal case of "log(|Sigma_1|/|Sigma_0|)"


#
# ~~~ From a torch.distributions Distribution class, define a method that samples from that standard distribution
class InverseTransformSampler:
    def __init__(self, icdf, generator=None):
        self.icdf = icdf
        self.generator = generator

    def __call__(self, *shape, device="cpu", dtype=torch.float):
        U = torch.rand(*shape, generator=self.generator, device=device, dtype=dtype)
        return self.icdf(U)


#
# ~~~ Compute Jacobian of a model's flattened outputs (at inputs) with respect to its parameters (*not* with respect to the inputs)
def compute_Jacobian_of_flattened_model(model, inputs, out_features):
    #
    # ~~~ Compute the Jacobian of model outputs with respect to model parameters
    J_dict = jacrev(functional_call, argnums=1)(
        model, dict(model.named_parameters()), (inputs,)
    )
    return torch.column_stack(
        [
            tens.reshape(
                out_features * len(inputs), -1
            )  # ~~~ trial and error led me here; not sure how well (or not) this use of `reshape` generalizes to other network architectures
            for tens in J_dict.values()
        ]
    )  # ~~~ has shape ( n_meas*out_features, n_params ) where n_params is the total number of weights/biases in a network of this architecture


#
# ~~~ Draw n random samples from multivariate normal distributions (mvns), of which k are provided (in the form of their mean and Sigma^{1/2}), each m-dimensional
def randmvns(mu, root_Sigma, n=1, **kwargs):
    m, k = mu.shape
    assert root_Sigma.shape == (k, m, m)
    IID_standard_normal_samples = torch.randn(k, m, n, device=mu.device, dtype=mu.dtype)
    #
    # ~~~ Sample from the N(mu,Sigma) distribution by taking mu + Sigma^{1/2}z, where z is a sampled from the N(0,I) distribtion
    return mu + torch.bmm(root_Sigma, IID_standard_normal_samples).permute(
        2, 1, 0
    )  # ~~~ returns a shape consistent with the output of `forward` and the assumption bnns.metrics: ( n_samples, n_test, n_out_features ), i.e., ( n, x.shape[0], self.out_features )


### ~~~
## ~~~ Dependencies from https://github.com/ThomasLastName/quality-of-life/blob/main/quality_of_life/my_numpy_utils.py
### ~~~

#
# ~~~ Compute the moving average of a list (this reduces the list's length)
moving_average = lambda list, window_size: np.convolve(
    list, np.ones(window_size) / window_size, mode="valid"
)
