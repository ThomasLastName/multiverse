
from abc import abstractmethod

import math
import torch
from torch import nn

from bnns.NoPriorBNNs import ConventionalVariationalFamilyBNN
from bnns.GPR import simple_mean_zero_RPF_kernel_GP

from quality_of_life.my_base_utils import my_warn



### ~~~
## ~~~ Implement `prior_forward` and `set_prior_hyperparameters` for a GP prior
### ~~~

class GPPriorBNN(ConventionalVariationalFamilyBNN):
    def __init__(
                #
                # ~~~ Architecture and stuff
                self,
                *args,
                likelihood_std = torch.tensor(0.001),
                auto_projection = True,
                #
                # ~~~ Specify the location-scale family of the variational distribution
                posterior_standard_log_density, # ~~~ should be a callable that accepts generic torch.tensors as input but also works on numpy arrays, e.g. `lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) )` for Gaussian
                posterior_standard_initializer, # ~~~ should modify its argument's `data` attribute in place and return None
                posterior_standard_sampler,     # ~~~ should return a tensor of random samples from the reference distribution
        ):
        super().__init__(
                *args,
                likelihood_std = likelihood_std,
                auto_projection = auto_projection,
                posterior_standard_log_density = posterior_standard_log_density,
                posterior_standard_initializer = posterior_standard_initializer,
                posterior_standard_sampler     = posterior_standard_sampler
            )
        self.default_bw = None          # ~~~ the median distance between training data is used if `None`
        self.default_prior_scale = None # ~~~ a list of all 1.'s is used if `None`
        self.default_eta = 0.001        # ~~~ add eta*I to the covariance matrices in the GP for numerical stability
        self.set_prior_hyperparameters( bw=self.default_bw, prior_scale=self.defaul_prior_scale, eta=self.default_eta )
    #
    # ~~~ Allow these to be set at runtime
    def set_prior_hyperparameters( self, **kwargs ):
        #
        # ~~~ If any of the 3 hyper-parameters bw, scale, or eta are unspecified, then use the class level defaults
        try:
            bw = kwargs["bw"]
        except KeyError:
            bw = self.default_bw
            my_warn(f'Hyper-parameter "bw" not specified. Using default value of {self.default_bw}.')
        try:
            prior_scale = kwargs["prior_scale"]
        except KeyError:
            prior_scale = self.default_prior_scale
            my_warn(f'Hyper-parameter "prior_scale" not specified. Using default value of {self.default_prior_scale}.')
        try:
            eta = kwargs["eta"]
        except KeyError:
            eta = self.default_eta
            my_warn(f'Hyper-parameter "eta" not specified. Using default value of {self.default_eta}.')
        #
        # ~~~ Define a mean zero RBF kernel GP with independent output channels all sharing the same value bw, scale, and eta
        self.GP = simple_mean_zero_RPF_kernel_GP( out_features=self.out_features, bw=bw, prior_scale=prior_scale, eta=eta )
    #
    # ~~~ Get the mean and sqare root of the covariance matrix of the GP from each output feature
    @abstractmethod
    def mean_and_sqrt_Sigma(self,x):
        raise NotImplementedError("For a ready-to-use implementation of a GP prior with zero mean and RBF kernel, please see the sub-class `SimpleGPPriorBNN`.")
    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just sample from the normal distribution with mean and covariance specified by the GP)
    def prior_forward( self, x, n=1 ):
        mu, root_Sigma = self.GP.prior_mu_and_Sigma( x, cholesky=True )  # ~~~ return the cholesky square roots of the covariance matrices
        Z = torch.randn( size=( n, x.shape[0] ), device=x.device, dtype=x.dtype )
        SZ = root_Sigma@Z.T # ~~~ SZ[:,:,j] is the matrix you get from stacking the vectors Sigma_i^{1/2}z_j for i=1,...,self.out_features, where z_j==Z[j] and Sigma_i is the covariance matrix of the model's i-th output
        #
        # ~~~ Sample from the N(mu,Sigma) distribution by taking mu+Sigma^{1/2}z, where z is a sampled from the N(0,I) distribtion
        return torch.row_stack([ (mu + SZ[:,:,j].T).flatten() for j in range(n) ])



### ~~~
## ~~~ Implement what I call the "Gaussian approximation method" of Rudner et al. 2023 (https://arxiv.org/abs/2312.17199), which assumes that the network weights are normally distributed
### ~~~

class GPPrior2023BNN(GPPriorBNN):
    def __init__(
                self,
                *args,
                likelihood_std = torch.tensor(0.001),
                auto_projection = True
            ):
        super().__init__(
                *args,
                likelihood_std = likelihood_std,
                auto_projection = auto_projection,
                posterior_standard_log_density = lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) ),
                posterior_standard_initializer = nn.init.normal_,
                posterior_standard_sampler     = torch.randn,
        )
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the fBNN loss using a Gaussian approximation (https://arxiv.org/abs/2312.17199)
    ### ~~~
    #
    # ~~~ Compute the mean and standard deviation of a normal distribution approximating q_theta
    def gaussian_kl( self, resample_measurement_set=True, add_stabilizing_noise=True, approximate_mean=False ):
        #
        # ~~~ If the variational farmily on the weights is not independent normal, then the formula for the KL divergence is not valid
        if (not self.passed_normal_test) and (not self.already_warned_about_normal_test):
            my_warn("The formula for the KL divergence implemented currently is invalid when the weights are not modeled as independent normal.")
            self.already_warned_about_normal_test = True
        #
        # ~~~ Get the mean and covariance of (the Gaussian approximation of) the predicted distribution of yhat
        mu_theta, Sigma_theta = self.mean_and_covariance_of_first_order_approximation( resample_measurement_set=resample_measurement_set, approximate_mean=approximate_mean )
        if add_stabilizing_noise:
            Sigma_theta += torch.diag( self.post_approximation_eta * torch.ones_like(Sigma_theta.diag()) )
        root_Sigma_theta = torch.linalg.cholesky(Sigma_theta)
        #
        # ~~~ Get the mean and covariance of the prior distribution of yhat (a Gaussian process)
        mu_0, root_Sigma_0 = self.GP.prior_mu_and_Sigma( self.measurement_set, inv=False, flatten=True, cholesky=True )
        Sigma_0_inv = torch.cholesky_inverse(root_Sigma_0)
        #
        # ~~~ Apply a formula for the KL divergence KL( N(mu_theta,Sigma_theta) || N(mu_0,Sigma_0) ); see `scripts/gaussian_kl_computations.py`
        return (
            (Sigma_0_inv@Sigma_theta).diag().sum()
            - len(mu_0)
            + torch.inner( mu_0-mu_theta, (Sigma_0_inv @ (mu_0-mu_theta)) )
            + 2*root_Sigma_0.diag().log().sum() - 2*root_Sigma_theta.diag().log().sum()
        ) / 2
