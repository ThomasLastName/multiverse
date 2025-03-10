
import torch

from bnns.NoPriorBNNs import IndepLocScaleSequentialBNN
from bnns.GPR import simple_mean_zero_RPF_kernel_GP

from quality_of_life.my_base_utils import my_warn



### ~~~
## ~~~ Implement `prior_forward` and `set_prior_hyperparameters` for a GP prior
### ~~~

class GPPriorBNN(IndepLocScaleSequentialBNN):
    def __init__(
                self,
                *args,
                prior_generator = None, # ~~~ the only new kwarg that this sub-class introduces
                **kwargs
        ):
        super().__init__( *args, **kwargs )
        #
        # ~~~ Set default values for hyper-parameters of the prior
        self.default_bw = None      # ~~~ the median distance between training data is used if `None`
        self.default_scale = None   # ~~~ a list of all 1.'s is used if `None`
        self.default_eta = 0.001    # ~~~ add eta*I to the covariance matrices in the GP for numerical stability
        self.set_prior_hyperparameters( bw=self.default_bw, scale=self.default_scale, eta=self.default_eta )
        self.prior_generator = prior_generator
    #
    # ~~~ Allow the hyper-parameters of the prior distribution to be set at runtime
    def set_prior_hyperparameters( self, **kwargs ):
        #
        # ~~~ If any of the 3 hyper-parameters bw, scale, or eta are unspecified, then use the class level defaults
        try:
            bw = kwargs["bw"]
        except KeyError:
            bw = self.default_bw
            my_warn(f'Hyper-parameter "bw" not specified. Using default value of {self.default_bw}.')
        try:
            scale = kwargs["scale"]
        except KeyError:
            scale = self.default_scale
            my_warn(f'Hyper-parameter "scale" not specified. Using default value of {self.default_scale}.')
        try:
            eta = kwargs["eta"]
        except KeyError:
            eta = self.default_eta
            my_warn(f'Hyper-parameter "eta" not specified. Using default value of {self.default_eta}.')
        #
        # ~~~ Define a mean zero RBF kernel GP with independent output channels all sharing the same value bw, scale, and eta
        self.GP = simple_mean_zero_RPF_kernel_GP( out_features=self.out_features, bw=bw, scale=scale, eta=eta )
    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just sample from the normal distribution with mean and covariance specified by the GP)
    def prior_forward( self, x, n=1 ):
        #
        # ~~~ Return the cholesky square roots of the covariance matrices;
        mu, root_Sigma = self.GP.prior_mu_and_Sigma( x, cholesky=True )
        assert root_Sigma.shape == ( self.out_features, x.shape[0], x.shape[0] )
        assert mu.shape == ( x.shape[0], self.out_features )
        IID_standard_normal_samples = torch.randn( self.out_features,x.shape[0],n, generator=self.prior_generator, device=x.device, dtype=x.dtype )
        #
        # ~~~ Sample from the N(mu,Sigma) distribution by taking m u +Sigma^{1/2}z, where z is a sampled from the N(0,I) distribtion
        return mu + torch.bmm( root_Sigma, IID_standard_normal_samples ).permute(2,1,0) # ~~~ returns a shape consistent with the output of `forward` and the assumption bnns.metrics: ( n_samples, n_test, n_out_features ), i.e., ( n, x.shape[0], self.out_features )
    #
    # ~~~ If using projected gradient descent, then project onto the non-negative orthant
    def apply_hard_projection( self, tol=1e-6 ):
        with torch.no_grad():
            for p in self.posterior_std.parameters():
                p.data.clamp_(min=tol)
    #
    # ~~~ If not using projected gradient descent, then "parameterize the standard deviation pointwise" such that any positive value is acceptable (as on page 4 of https://arxiv.org/pdf/1505.05424)
    def setup_soft_projection( self, method="Blundell" ):
        if method == "Blundell":
            self.soft_projection = lambda x: torch.log( 1 + torch.exp(x) )
            self.soft_projection_inv = lambda x: torch.log( torch.exp(x) - 1 )
            self.soft_projection_prime = lambda x: 1 / (1 + torch.exp(-x))
        elif method == "torchbnn":
            self.soft_projection = lambda x: torch.exp(x)
            self.soft_projection_inv = lambda x: torch.log(x)
            self.soft_projection_prime = lambda x: torch.exp(x)
        else:
            raise ValueError(f'Unrecognized method="{method}". Currently, only method="Blundell" and "method=torchbnn" are supported.')
    #
    # ~~~ If using projected gradient descent, then project onto the non-negative orthant
    def apply_soft_projection(self):
        with torch.no_grad():
            for p in self.posterior_std.parameters():
                p.data = self.soft_projection(p.data)



### ~~~
## ~~~ Implement what I call the "Gaussian approximation method" of Rudner et al. 2023 (https://arxiv.org/abs/2312.17199), which assumes that the network weights are normally distributed
### ~~~

class GPPrior2023BNN(GPPriorBNN):
    def __init__( self, *args, post_approximation_eta=0.01, **kwargs ):
        super().__init__( *args, **kwargs )
        self.post_approximation_eta = post_approximation_eta
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the fBNN loss using a Gaussian approximation (https://arxiv.org/abs/2312.17199)
    ### ~~~
    #
    # ~~~ Compute the mean and standard deviation of a normal distribution approximating q_theta
    def gaussian_kl( self, resample_measurement_set=True, add_stabilizing_noise=True, approximate_mean=False ):
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
