
from abc import abstractmethod

import torch
from torch import nn
from torch.func import jacrev, functional_call

from bnns.utils import manual_Jacobian, flatten_parameters, std_per_param, std_per_layer
from bnns.SSGE import SpectralSteinEstimator as SSGE
from bnns.utils import log_gaussian_pdf

from quality_of_life.my_base_utils import my_warn
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list



### ~~~
## ~~~ Define a very broad BNN which does little more than implement SSGE
### ~~~

#
# ~~~ Main class: intended to mimic nn.Module
class BayesianModule(nn.Module):
    def __init__(self):
        super().__init__()
        #
        # ~~~ Attributes for SSGE and functional training
        self.prior_J   = "please specify"
        self.post_J    = "please specify"
        self.prior_eta = "please specify"
        self.post_eta  = "please specify"
        self.prior_M   = "please specify"
        self.post_M    = "please specify"
        self.prior_SSGE = None
    #
    # ~~~ Resample from whatever is source is used to seed the samples which are drawn from the variational distribution
    @abstractmethod
    def resample_weights(self):
        raise NotImplementedError("The base class BayesianModule leaves the `resample_weights` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return a sample from the "variationally distributed" (i.e., learned) outputs of the network; this is like f(x;w) where w is sampled from a varitaional (i.e., learned) distribution over network weights
    @abstractmethod
    def forward( self, x, resample_weights=True ):
        raise NotImplementedError("The base class BayesianModule leaves the `forward` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an estimate of `\int ln(f_{Y \mid X,W}(w,X,y)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta`, and `f_{Y \mid X,W}(w,X,y)` is the likelihood density
    @abstractmethod
    def estimate_expected_log_likelihood(self,X,y):
        raise NotImplementedError("The base class BayesianModule leaves the `estimate_expected_log_likelihood` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the loss in Bayes by Backprop
    ### ~~~
    #
    # ~~~ Return the exact kl divergence between the variational distribution and a prior distribution over weights, if applicable
    @abstractmethod
    def exact_weight_kl(self):
        raise NotImplementedError("The base class BayesianModule leaves the `exact_weight_kl` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an esimate of `\int ln(q_\theta(w)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta`
    @abstractmethod
    def estimate_expected_log_posterior(self):
        raise NotImplementedError("The base class BayesianModule leaves the `estimate_expected_log_posterior` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an esimate of `\int \ln(f_W(w)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta`, and `f_W(w)` is a prior density function over network weights
    @abstractmethod
    def estimate_expected_log_prior(self):
        raise NotImplementedError("The base class BayesianModule leaves the `estimate_expected_log_prior` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an estimate (or the exact value) of the kl divergence between variational and prior distributions over newtork weights
    def weight_kl( self, exact_formula=True ):
        self.resample_weights()
        if exact_formula:
            try:
                return self.exact_weight_kl()
            except NotImplementedError:
                if not hasattr(self,"already_warned_that_exact_weight_formula_not_implemented"):
                    my_warn("`exact_weight_kl(exact_formula=True)` method raised a NotImplementedError; will fall back to using `weight_kl(exact_formula=False)` instead.")
                    self.already_warned_that_exact_weight_formula_not_implemented = "yup"
            except:
                raise
        estimate_of_weight_kl =  self.estimate_expected_log_posterior() - self.estimate_expected_log_prior()
        self.kl_was_just_computed = True
        return estimate_of_weight_kl
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the fBNN loss from Sun et al. 2019 (https://arxiv.org/abs/1903.05779)
    ### ~~~
    #
    # ~~~ Sample from the priorly distributed outputs of the network
    @abstractmethod
    def prior_forward( self, x, n=1 ):
        raise NotImplementedError("The base class BayesianModule leaves the `prior_forward` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Generate a fresh grid of several "points like our model's inputs" from the input domain
    @abstractmethod
    def sample_new_measurement_set(self,n=64):
        raise NotImplementedError("The base class BayesianModule leaves the `sample_new_measurement_set` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Instantiate the SSGE estimator of the prior score, using samples from the prior distribution
    def setup_prior_SSGE(self):
        with torch.no_grad():
            #
            # ~~~ Sample from the prior distribution self.prior_M times (and flatten the samples)
            prior_samples = self.prior_forward( self.measurement_set, n=self.prior_M )
            #
            # ~~~ Build an SSGE estimator using those samples
            try:
                #
                # ~~~ First, try the implementation of the linalg routine using einsum
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.prior_eta, J=self.prior_J )
            except:
                #
                # ~~~ In case that crashes due to not enough RAM, then try the more memory-efficient (but slower) impelemntation of the same routine using a for loop
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.prior_eta, J=self.prior_J, iterative_avg=True )
    #
    # ~~~ Estimate KL_div( variational output || prior output ) using the SSGE, assuming we don't have a forula for the density of the variational distribution of the outputs
    def functional_kl( self, resample_measurement_set=True, return_raw_ingrdients=False ):
        #
        # ~~~ If `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
        #
        # ~~~ Prepare for using SSGE to estimate some of the gradient terms
        with torch.no_grad():
            if (self.prior_SSGE is None) or resample_measurement_set:
                self.setup_prior_SSGE()
            posterior_samples = torch.row_stack([ self(self.measurement_set).flatten() for _ in range(self.post_M) ])
            if posterior_samples.std(dim=0).max()==0:
                raise ValueError("The posterior samples have zero variance. This is likely because the forward method neglects to resample weights from the variational distribution.")
            posterior_SSGE = SSGE( samples=posterior_samples, eta=self.post_eta, J=self.post_J )
        #
        # ~~~ By the chain rule, at these points we must compute the "scores," i.e., gradients of the log-densities (we use SSGE to compute them)
        yhat = self(self.measurement_set).flatten()
        #
        # ~~~ Use SSGE to compute "the intractible parts of the chain rule"
        with torch.no_grad():
            posterior_score_at_yhat =  posterior_SSGE( yhat.reshape(1,-1) )
            prior_score_at_yhat     = self.prior_SSGE( yhat.reshape(1,-1) )
        #
        # ~~~ For generality, add this option that I never intend to use
        if return_raw_ingrdients:
            return yhat, posterior_score_at_yhat, prior_score_at_yhat
        #
        # ~~~ Combine all the ingridents as per the chain rule 
        estimate_of_log_posterior_expectation = ( posterior_score_at_yhat @ yhat ).squeeze()  # ~~~ the inner product from the chain rule
        estimate_of_log_prior_expectation     = ( prior_score_at_yhat @ yhat ).squeeze()      # ~~~ the inner product from the chain rule            
        self.kl_was_just_computed = True
        return estimate_of_log_posterior_expectation - estimate_of_log_prior_expectation



### ~~~
## ~~~ Define a BNN with the necessary methods
### ~~~

#
# ~~~ Main class: intended to mimic nn.Sequential (STILL NO PRIOR DISTRIBUTION AT THIS LEVEL OF ABSTRACTION)
class IndependentLocationScaleSequentialBNN(BayesianModule):
    def __init__(
                self,
                *args,
                model_log_density,  # ~~~ should be a function of `where`, `mu`, and `sigma`
                model_initializer,  # ~~~ should modify its argument's `data` attribute in place and return None
                conditional_std = torch.tensor(0.001),
                auto_projection = True
            ):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        self.model_mean = nn.Sequential(*args)
        self.model_std  = nonredundant_copy_of_module_list( self.model_mean, sequential=True )
        if auto_projection:
            self.ensure_positive( forceful=True, verbose=False )
        #
        # ~~~ Basic information about the model: in_features, out_features, and n_layers
        self.n_layers = len(self.model_mean)
        for layer in self.model_mean:
            if hasattr(layer,"in_features"):    # ~~~ the first layer with an `in_features` attribute
                self.in_features = layer.in_features 
                break
        for layer in reversed(self.model_mean):
            if hasattr(layer,"out_features"):   # ~~~ the last layer with an `out_features` attribute
                self.out_features = layer.out_features
                break
        #
        # ~~~ Define a "standard normal [or whatever else] distribution in the shape of our neural network"
        self.model_log_density = model_log_density
        self.model_initializer = model_initializer # ~~~ this is "ducky" (https://en.wikipedia.org/wiki/Duck_typing); can be anything that modifies the input's `data` attribute in place
        self.realized_standard_distribution = nonredundant_copy_of_module_list(self.model_mean)
        self.sample_from_standard_distribution()
        #
        # ~~~ Attributes determining the log likelihood density
        self.likelihood_model = "Gaussian"
        self.conditional_std = conditional_std
        #
        # ~~~ Attributes used for testing validity of the default measurement set
        self.first_moments_of_input_batches = []
        self.second_moments_of_input_batches = []
        #
        # ~~~ A flag used for warning of a easily occuring possible failure case
        self.kl_was_just_computed = False
    # ~~~
    #
    ### ~~~
    ## ~~~ Basic methods such as "check that the weights are positive" (`ensure_positive`) and "make the weights positive" (`apply_hard_projection` and `soft_projection`)
    ### ~~~
    #
    # ~~~ Infer device and dtype
    def infer_device_and_dtype(self):
        for layer in self.model_mean:
            if hasattr(layer,"weight"):         # ~~~ the first layer with weights
                device = layer.weight.device
                dtype = layer.weight.dtype
                return device, dtype
    #
    # ~~~ Sample according to a "standard normal [or other] distribution in the shape of our neural network"
    def sample_from_standard_distribution(self):
        with torch.no_grad():   # ~~~ theoretically the `no_grad()` context is redundant and unnecessary, but idk why not use it
            for p in self.realized_standard_distribution.parameters():
                self.model_initializer(p)
    #
    # ~~~
    def resample_weights(self):
        self.sample_from_standard_distribution()

    #
    # ~~~ Check that all the posterior standard deviations are positive
    def ensure_positive( self, forceful=False, verbose=False ):
        with torch.no_grad():
            if not flatten_parameters(self.model_std).min() >= 0:
                #
                # ~~~ If an attribute `soft_projection` is defined, assume that the user simply forgot to use it
                if hasattr(self,"soft_projection") or verbose:
                    my_warn("`model_std` contains negative values.")
                #
                # ~~~ This is fine to use even when a soft_projection is intended, since `apply_hard_projection` is assumed to leave any values that are already in the desired range unaffected
                if forceful:
                    self.apply_hard_projection()
    #
    # ~~~ Whatever constraints we want the standard deviations to satisfy, implement a projection onto the constraint set such that apply_hard_projection(sigma)==sigma if sigma already already satisfies the constraints
    @abstractmethod
    def apply_hard_projection( self, tol=1e-6 ):
        with torch.no_grad():
            raise NotImplementedError("The class IndependentLocationScaleSequentialBNN leaves the method `apply_hard_projection` to be implented in user-defined subclasses, because it may depend on the prior distribution.")
    #
    # ~~~ Multiply parameter gradients by the transpose of the Jacobian of `soft_projection` (as in Blundell et al. 2015 https://arxiv.org/abs/1505.05424, where the Jacobian is diagonal and you just simply divide by 1+exp(-rho) )
    def apply_chain_rule_for_soft_projection(self):
        with torch.no_grad():
            for p in self.model_std.parameters():
                p.data = self.soft_projection_inv(p.data)           # ~~~ now, the parameters are \soft_projection = \ln(\exp(\sigma)-1) instead of \sigma
                try:
                    p.grad *= self.soft_projection_prime(p.data)    # ~~~ now, the gradient is \frac{\sigma'}{1+\exp(-\rho)} instead of \sigma'
                except:
                    if p.grad is None:
                        my_warn("`apply_chain_rule_for_soft_projection` operates directly on the `grad` attributes of the parameters. It should be applied *after* `backwards` is called.")
                    raise
    #
    # ~~~ Initialize the posterior standard deviations to match the standard deviations of a possible prior distribution
    def set_default_uncertainty( self, comparable_to_default_torch_init=False, scale=1.0 ):
        with torch.no_grad():
            if comparable_to_default_torch_init:
                for layer in self.model_std:
                    if isinstance(layer,nn.Linear):
                        std = std_per_layer(layer)
                        layer.weight.data = std * torch.ones_like(layer.weight.data)
                        if layer.bias is not None:
                            layer.bias.data = std * torch.ones_like(layer.bias.data)
            else:
                for p in self.model_std.parameters():
                    p.data = std_per_param(p)*torch.ones_like(p.data)
            #
            # ~~~ Scale the parameters of the last linear layer; for sequential models, the effect is comparable to the scale paramter in a GP
            if scale is not None:
                for layer in reversed(self.model_std):
                    if isinstance( layer, nn.Linear ):
                        for p in layer.parameters():
                            p.data *= scale
                        break
    #
    # ~~~ Sample the distribution of Y|X=x,W=w
    def forward( self, x, resample_weights=True ):
        #
        # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_distribution
        if resample_weights:
            self.sample_from_standard_distribution()      # ~~~ this method re-generates the values of weights and biases in `self.realized_standard_distribution` (IID standard normal)
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        self.ensure_positive(forceful=True)
        for j in range(self.n_layers):
            z = self.realized_standard_distribution[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            #
            # ~~~ If this layer is just like relu or something, then there aren't any weights; just apply the layer and be done
            if not isinstance( z, nn.Linear ):
                x = z(x)                            # ~~~ x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.model_mean[j]     # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer  =  self.model_std[j]     # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                A = mean_layer.weight + std_layer.weight * z.weight # ~~~ A = F_\theta(z.weight) is normal with the trainable (posterior) mean and std
                x = x@A.T                                           # ~~~ apply the appropriately distributed weights to this layer's input
                if z.bias is not None:
                    x = x + (mean_layer.bias + std_layer.bias * z.bias) # ~~~ apply the appropriately distributed biases
        return x
    #
    # ~~~ Compute ln( f_{Y \mid X,W}(F_\theta(z),x_train,y_train) ) at a point z sampled from the standard MVN distribution ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def estimate_expected_log_likelihood( self, X, y, use_input_in_next_measurement_set=False, verbose=True ):
        #
        # ~~~ Store the input itself, and/or descriptive statistics, for reference when generating the measurement set
        self.first_moments_of_input_batches.append(X.mean(dim=0))
        self.second_moments_of_input_batches.append((X**2).mean(dim=0))
        if use_input_in_next_measurement_set:
            self.desired_measurement_points = X
        #
        # ~~~ Attempt to warn about a common pitfall
        if verbose and not self.kl_was_just_computed:
            my_warn("It seems that `estimate_expected_log_likelihood` was called BEFORE the KL divergence was estimated. For at least one training method (viz. Sun et al. 2019), this is nigh impossible to support (due to the need to generate fresh samples to construct the SSGE, entailing inplace modification of variables used in the graph computational created by `estimate_expected_log_likelihood`). Therefore, `bnns` refuses to support it entirely. Please do not call `estimate_expected_log_likelihood` until AFTER the KL divergence has been estimated.")
        #
        # ~~~ The likelihood depends on task criterion: classification or regression
        self.ensure_positive(forceful=True)
        if self.likelihood_model == "Gaussian":
            log_lik = log_gaussian_pdf( where=y, mu=self(X,resample_weights=False), sigma=self.conditional_std )    # ~~~ Y|X,W is assumed to be normal with mean self(X) and variance self.conditional_std (the latter being a tunable hyper-parameter)
        else:
            raise NotImplementedError("In the current version of the code, only the Gaussian likelihood (i.e., mean squared error) is implemented See issue ?????.")
        self.kl_was_just_computed = False
        return log_lik
    # ~~~
    #
    ### ~~~
    ## ~~~ Method for computing the loss in Bayes by Backprop
    ### ~~~
    #
    # ~~~ Compute \ln( q_\theta(F_\theta(z)) ) at a point z sampled from the standard MVN distribution, where q_\theta is the posterior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def estimate_expected_log_posterior(self):
        self.ensure_positive(forceful=True)
        mu_post = flatten_parameters(self.model_mean)
        sigma_post = flatten_parameters(self.model_std)
        z_sampled = flatten_parameters(self.realized_standard_distribution)
        w_sampled = mu_post + sigma_post*z_sampled
        return self.model_log_density( where=w_sampled, mu=mu_post, sigma=sigma_post )
    # ~~~
    #
    ### ~~~
    ## ~~~ Method for computing the fBNN loss from Rudner et al. 2023 (https://arxiv.org/abs/2312.17199)
    ### ~~~
    #
    # ~~~ Compute the mean and standard deviation of a distribution approximating q_theta
    def mean_and_covariance_of_first_order_approximation( self, resample_measurement_set=True, approximate_mean=False ):
        #
        # ~~~ If `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
        #
        # ~~~ Assume that the final layer of the architecture is linear, as per the paper's suggestion to take \beta as the parameters of the final layer (very bottom of pg. 4 https://arxiv.org/pdf/2312.17199)
        if not isinstance( self.model_mean[-1] , nn.Linear ):
            raise NotImplementedError('Currently, the only case implemented is the the case from the paper where `beta` is "the set of parameters in the final neural network layer" (bottom of pg. 4 of the paper). Moreover, this is only implemented when the final layer has a bias term.')
        elif self.model_mean[-1].bias is None:
            raise NotImplementedError('Currently, the only case implemented is the the case from the paper where `beta` is "the set of parameters in the final neural network layer" (bottom of pg. 4 of the paper). Moreover, this is only implemented when the final layer has a bias term.')
        #
        # ~~~ Compute the mean and covariance of a normal distribution approximating that of the (random) output of the network on the measurement set
        self.ensure_positive(forceful=True)
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
    # ~~~ In the common case that the inputs are standardized, then standard random normal vectors are "points like our model's inputs"
    def sample_new_measurement_set( self, n=64, after_how_many_batches_to_warn=100, tol=0.25 ):
        #
        # ~~~ Attempt to assess validity of this default implementaiton
        if not isinstance( self.model_mean[0], nn.Linear ):
            my_warn("Because the first model layer is not a linear layer, the default implementation of `sample_new_measurement_set` may fail. If so (or to avoid this warning message), please sub-class the model you wish to use and implement sample_new_measurement_set() for the sub-class.")
        if len(self.first_moments_of_input_batches)==after_how_many_batches_to_warn:   # ~~~ warn only once, with a sample size of 100
            estimated_mean_of_all_inputs = torch.stack(self.first_moments_of_input_batches).mean(dim=0).max()
            estimated_var_of_all_inputs  = torch.stack(self.second_moments_of_input_batches).mean(dim=0).max() - estimated_mean_of_all_inputs**2  # ~~~ var(X) = E(X^2) - E(X)^2
            if estimated_mean_of_all_inputs.abs()>tol or estimated_var_of_all_inputs>1+tol:
                my_warn("the default implementation of `sample_new_measurement_set` assumes inputs are N(0,1) however this assumption appears to be violated. Please consider programming a data-specific implementation of `sample_new_measurement_set` for better results.")
        #
        # ~~~ Do the default implementation
        device, dtype = self.infer_device_and_dtype()
        if hasattr(self,"desired_measurement_points"):
            batch_size = len(self.desired_measurement_points)
            if batch_size>n:
                my_warn("More desired measurement points are specified than the total number of measurement points (this is most likely the result batch size exceeding the specified number of measurement points). Only a randomly chosen subset of the desired measurement points will be used.")
                self.measurement_set = self.desired_measurement_points[torch.randperm(batch_size)[:n]]
            else:
                self.measurement_set = torch.vstack([
                        self.desired_measurement_points,
                        torch.randn( n-batch_size, self.in_features, device=device, dtype=dtype )
                    ])
        else:
            self.measurement_set = torch.randn( size=(n,self.in_features), device=device, dtype=dtype )



### ~~~
## ~~~ Implement the projection methods assuming that any positive variacne, and any mean at all are acceptable
### ~~~

class ConventionalSequentialBNN(IndependentLocationScaleSequentialBNN):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs)
    #
    # ~~~ If not using projected gradient descent, then "parameterize the standard deviation pointwise" such that any positive value is acceptable (as on page 4 of https://arxiv.org/pdf/1505.05424)
    def setup_soft_projection( self, method="Blundell" ):
        if method=="Blundell":
            self.soft_projection = lambda x: torch.log( 1 + torch.exp(x) )
            self.soft_projection_inv = lambda x: torch.log( torch.exp(x) - 1 )
            self.soft_projection_prime = lambda x: 1 / (1 + torch.exp(-x))
        elif method=="torchbnn":
            self.soft_projection = lambda x: torch.exp(x)
            self.soft_projection_inv = lambda x: torch.log(x)
            self.soft_projection_prime = lambda x: torch.exp(x)
        else:
            raise ValueError(f'Unrecognized method="{method}". Currently, only method="Blundell" and "method=torchbnn" are supported.')
    #
    # ~~~ If using projected gradient descent, then project onto the non-negative orthant
    def apply_hard_projection( self, tol=1e-6 ):
        with torch.no_grad():
            for p in self.model_std.parameters():
                p.data.clamp_(min=tol)
    #
    # ~~~ If using projected gradient descent, then project onto the non-negative orthant
    def apply_soft_projection(self):
        with torch.no_grad():
            for p in self.model_std.parameters():
                p.data = self.soft_projection(p.data)
