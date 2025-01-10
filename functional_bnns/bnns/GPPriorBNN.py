
from abc import abstractmethod

import torch
from torch.func import jacrev, functional_call

from bnns.utils import manual_Jacobian
from bnns.SequentialGaussianBNN import IndependentLocationScaleBNN, flatten_parameters
from bnns.GPR import simple_mean_zero_RPF_kernel_GP

from quality_of_life.my_base_utils import my_warn


#
# ~~~ Implement `prior_forward` for a GP prior, and implement what I call the "Gaussian approximation method" of Rudner et al. 2023 (https://arxiv.org/abs/2312.17199)
class GPPriorBNN(IndependentLocationScaleBNN):
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
    def simple_gaussian_approximation( self, resample_measurement_set=True, approximate_mean=False ):
        #
        # ~~~ if `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
        #
        # ~~~ Assume that the final layer of the architecture is linear, as per the paper's suggestion to take \beta as the parameters of the final layer (very bottom of pg. 4 https://arxiv.org/pdf/2312.17199)
        if not isinstance( self.model_mean[-1] , torch.nn.modules.linear.Linear ):
            raise NotImplementedError('Currently, the only case implemented is the the case from the paper where `\beta` is "the set of parameters in the final neural network layer" (bottom of pg. 4 of the paper).')
        #
        # ~~~ Compute the mean and covariance of a normal distribution approximating that of the (random) output of the network on the measurement set
        n_meas = self.measurement_set.shape[0]
        out_features = self.model_mean[-1].out_features
        if not approximate_mean:
            #
            # ~~~ Compute the mean and covariance from the paper's equation (14): https://arxiv.org/abs/2312.17199, page 4
            S_sqrt = torch.cat([ p.flatten() for p in self.model_std.parameters() ])  # ~~~ covariance of the joint (diagonal) normal distribution of all network weights is then S_sqrt.diag()**2
            theta_minus_m = S_sqrt*flatten_parameters(self.realized_standard_distribution)            # ~~~ theta-m == S_sqrt*z because theta = m+S_sqrt*z
            J_dict = jacrev( functional_call, argnums=1 )(
                    self.model_mean,
                    dict(self.model_mean.named_parameters()),
                    (self.measurement_set,)
                )
            full_Jacobian = torch.column_stack([
                    tens.reshape( out_features*n_meas, -1 )
                    for tens in J_dict.values()
                ])  # ~~~ has shape ( n_meas*out_features, n_params ) where n_params is the total number of weights/biases in a network of this architecture
            how_many_params_from_not_last_layer = len(flatten_parameters( self.model_mean[:-1] ))
            J_alpha = full_Jacobian[ :, :how_many_params_from_not_last_layer ]              # ~~~ Jacobian with respect to parameters in not the last layer, same as what the paper calls J_\alpha
            J_beta = full_Jacobian[ :, how_many_params_from_not_last_layer: ]               # ~~~ Jacobian with respect to parameters in only last layer,    same as what the paper calls J_\beta
            S_beta_sqrt = S_sqrt[ how_many_params_from_not_last_layer: ]                    # ~~~ our S_beta.diag()**2 is what the paper calls S_\beta (which is a diagonal matrix by design)
            theta_alpha_minus_m_alpha = theta_minus_m[:how_many_params_from_not_last_layer] # ~~~ same as what the paper calls theta_\alpha - m_\alpha
            mu_theta = self.model_mean(self.measurement_set).flatten() + J_alpha@theta_alpha_minus_m_alpha  # ~~~ mean from the paper's eq'n (14)
            Sigma_theta = (S_beta_sqrt*J_beta) @ (S_beta_sqrt*J_beta).T                                     # ~~~ cov. from the paper's eq'n (14)    
        if approximate_mean:
            #
            # ~~~ Only estimate the mean from the paper's eq'n (14), still computing the covariance exactly
            S_beta_sqrt = torch.cat([   # ~~~ the covaraince matrix of the weights and biases of the final layer is then S_beta_sqrt.diag()**2
                    self.model_std[-1].weight.flatten(),
                    self.model_std[-1].bias.flatten()
                ])
            z = torch.cat([             # ~~~ equivalent to `flatten_parameters(self.realized_standard_distribution)[ how_many_params_from_not_last_layer: ]`
                    self.realized_standard_distribution[-1].weight.flatten(),
                    self.realized_standard_distribution[-1].bias.flatten()
                ])
            theta_beta_minus_m_beta = S_beta_sqrt * z  # ~~~ theta_beta = mu_theta + Sigma_beta*z is sampled as theta_sampled = mu_theta + Sigma_theta*z_sampled (a flat 1d vector)
            #
            # ~~~ Jacbian w.r.t. the final layer's weights is easy to compute by hand: viz. the Jacobian of A@whatever w.r.t. A is, simply `whatever`; we first compute the `whatever` and then just shape it correctly
            whatever = self.model_mean[:-1](self.measurement_set)           # ~~~ just don't apply the final layer of self.model_mean
            J_beta = manual_Jacobian( whatever, out_features, bias=True )   # ~~~ simply shape it correctly
            #
            # ~~~ Deviate slightly from the paper by not actually computing J_alpha, and instead only approximating the requried sample
            mu_theta = self( self.measurement_set, resample_weights=False ).flatten() - J_beta @ theta_beta_minus_m_beta   # ~~~ solving for the mean of the paper's eq'n (14) by subtracting J_beta(theta_beta-m_beta) from the paper's equation (12)
            Sigma_theta = (S_beta_sqrt*J_beta) @ (S_beta_sqrt*J_beta).T
        return mu_theta, Sigma_theta
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
