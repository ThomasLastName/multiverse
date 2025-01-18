
from abc import abstractmethod

import torch

from bnns.NoPriorBNNs import ConventionalSequentialBNN
from bnns.GPR import simple_mean_zero_RPF_kernel_GP

from quality_of_life.my_base_utils import my_warn


#
# ~~~ Implement `prior_forward` for a GP prior, and implement what I call the "Gaussian approximation method" of Rudner et al. 2023 (https://arxiv.org/abs/2312.17199)
class GPPriorBNN(ConventionalSequentialBNN):
    def __init__(
            self,
            *args,
            family_log_density,
            family_initializer,
            conditional_std = torch.tensor(0.001),
            post_approximation_eta = 0.0001
        ):
        super().__init__(
                *args,
                family_log_density = family_log_density,
                family_initializer = family_initializer,
                conditional_std = conditional_std
            )
        #
        # ~~~ A hyperparameter needed for the gaussian appxroximation method
        self.post_eta = post_approximation_eta
        #
        # ~~~ Test for normalcy
        self.passed_normal_test = False # ~~~ TODO use scipy.stats.normaltest
        self.already_warned_about_normal_test = False
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
        mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=resample_measurement_set, approximate_mean=approximate_mean )
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

#
# ~~~ Define the abstract methods required for a GP prior
class SimpleGPPriorBNN(GPPriorBNN):
    def __init__(
            self,
            *args,
            family_log_density,
            family_initializer,
            prior_eta,
            prior_scale = 1.,
            post_approximation_eta = 0.0001,
            conditional_std = torch.tensor(0.001)
        ):
        super().__init__(
                *args,
                family_log_density = family_log_density,
                family_initializer = family_initializer,
                conditional_std = conditional_std,
                post_approximation_eta = post_approximation_eta
            )
        #
        # ~~~ Store a GP as an attribute using the package's built-in class
        self.GP = simple_mean_zero_RPF_kernel_GP(
            out_features = self.out_features,
            scale = prior_scale,
            eta = prior_eta
        )
    #
    # ~~~ Get the mean and sqare root of the covariance matrix of the GP from each output feature
    def mean_and_sqrt_Sigma(self,x):
        return self.GP.prior_mu_and_Sigma( x, cholesky=True ) 
