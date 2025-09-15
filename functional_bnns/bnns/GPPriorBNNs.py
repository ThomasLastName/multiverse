import torch

from bnns.utils.handling import my_warn
from bnns.NoPriorBNNs import FullSupportIndepLocScaleBNN
from bnns.GPR import GPYBackend, RPF_kernel_GP


### ~~~
## ~~~ Implement `prior_forward` and `set_prior_hyperparameters` for a GP prior
### ~~~


class GPPriorBNN(FullSupportIndepLocScaleBNN):
    def __init__(
        self,
        *args,
        prior_generator=None,  # ~~~ the only new kwarg that this sub-class introduces
        bw=None,
        scale=1,
        eta=0.001,
        gpytorch=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.set_prior_hyperparameters(bw=bw, scale=scale, eta=eta, gpytorch=gpytorch)
        self.prior_generator = prior_generator

    #
    # ~~~ Allow the hyper-parameters of the prior distribution to be set at runtime
    def set_prior_hyperparameters(self, bw, scale, eta, gpytorch):
        #
        # ~~~ Define a mean zero RBF kernel GP with independent output channels all sharing the same value bw, scale, and eta
        device, dtype = self.infer_device_and_dtype()
        fake_data_of_correct_shape = torch.randn(10, self.in_features).to(
            device=device, dtype=dtype
        )
        if gpytorch:
            if eta < 1e-4:
                my_warn(
                    f'The supplied value of eta={eta} is smaller than the lowerest value 1e-4 allowed by GPyTorch. You must, either, use a higher value of eta, or switch to the non-GPyTorch backend by specifying `"gpytorch" : false` as one of the prior hyper-parameters.'
                )
            self.GP = GPYBackend(
                x=fake_data_of_correct_shape,
                out_features=self.out_features,
                bws=self.out_features * [bw],
                scales=self.out_features * [scale],
                etas=self.out_features * [eta],
            )
        else:
            self.GP = RPF_kernel_GP(
                out_features=self.out_features,
                bws=self.out_features * [bw],
                scales=self.out_features * [scale],
                etas=self.out_features * [eta],
            )

    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just sample from the normal distribution with mean and covariance specified by the GP)
    def prior_forward(self, x, n=1):
        return self.GP.prior_forward(x, n)


### ~~~
## ~~~ Implement what I call the "Gaussian approximation method" of Rudner et al. 2023 (https://arxiv.org/abs/2312.17199), which assumes that the network weights are normally distributed
### ~~~


class GPPrior2023BNN(GPPriorBNN):
    def __init__(self, *args, post_approximation_eta=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_approximation_eta = post_approximation_eta

    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the fBNN loss using a Gaussian approximation (https://arxiv.org/abs/2312.17199)
    ### ~~~
    #
    # ~~~ Compute the mean and standard deviation of a normal distribution approximating q_theta
    def gaussian_kl(
        self,
        resample_measurement_set=True,
        add_stabilizing_noise=True,
        approximate_mean=False,
    ):
        #
        # ~~~ Get the mean and covariance of (the Gaussian approximation of) the predicted distribution of yhat
        mu_theta, Sigma_theta = self.mean_and_covariance_of_first_order_approximation(
            resample_measurement_set=resample_measurement_set,
            approximate_mean=approximate_mean,
        )
        if add_stabilizing_noise:
            Sigma_theta += torch.diag(
                self.post_approximation_eta * torch.ones_like(Sigma_theta.diag())
            )
        root_Sigma_theta = torch.linalg.cholesky(Sigma_theta)
        #
        # ~~~ Get the mean and covariance of the prior distribution of yhat (a Gaussian process)
        mu_0, root_Sigma_0 = self.GP.prior_mu_and_Sigma(
            self.measurement_set, flatten=True, cholesky=True
        )
        Sigma_0_inv = torch.cholesky_inverse(root_Sigma_0)
        #
        # ~~~ Apply a formula for the KL divergence KL( N(mu_theta,Sigma_theta) || N(mu_0,Sigma_0) ); see `scripts/gaussian_kl_computations.py`
        return (
            (Sigma_0_inv @ Sigma_theta).diag().sum()
            - len(mu_0)
            + torch.inner(mu_0 - mu_theta, (Sigma_0_inv @ (mu_0 - mu_theta)))
            + 2 * root_Sigma_0.diag().log().sum()
            - 2 * root_Sigma_theta.diag().log().sum()
        ) / 2
