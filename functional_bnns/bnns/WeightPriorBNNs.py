
import math
import torch
from torch import nn

from bnns.utils import flatten_parameters, log_gaussian_pdf, diagonal_gaussian_kl, std_per_param, std_per_layer, LocationScaleLogDensity
from bnns.NoPriorBNNs import ConventionalVariationalFamilyBNN

from quality_of_life.my_base_utils  import my_warn
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list



### ~~~
## ~~~ Implement `estimate_expected_prior_log_density`, `prior_forward`, and `set_prior_hyperparameters` for the "homoskedastic" mixture prior on the network weights employed in Blundell et al. 2015 (https://arxiv.org/abs/1505.05424)
### ~~~

class MixtureWeightPrior2015BNN(ConventionalVariationalFamilyBNN):
    def __init__(
                #
                # ~~~ Architecture and stuff
                self,
                *args,
                likelihood_std  = torch.tensor(0.001),
                auto_projection = True,
                #
                # ~~~ For the variational family, use "fully factored Gaussian weights" by default (to use another location-scale family, change these 3 argumetns)
                posterior_standard_log_density = lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) ),
                posterior_standard_sampler     = torch.randn,
            ):
        super().__init__(
                *args,
                likelihood_std  = likelihood_std,
                auto_projection = auto_projection,
                posterior_standard_log_density = posterior_standard_log_density,
                posterior_standard_sampler     = posterior_standard_sampler
            )
        #
        # ~~~ Set default values for hyper-parameters of the prior found here: https://github.com/danielkelshaw/WeightUncertainty/blob/master/torchwu/bayes_linear.py
        self.default_pi = torch.tensor(0.5) # ~~~ WARNING: this is not the mathematical constant pi\approx3.14. I don't appreciate Blundell et al.'s use of "\pi" to refer to a value between 0 and 1...
        self.default_sigma1 = torch.tensor(1.)
        self.default_sigma2 = torch.tensor(0.002)
        self.set_prior_hyperparameters( pi=self.default_pi, sigma1=self.default_sigma1, sigma2=self.default_sigma2 )
    #
    # ~~~ Set the hyper-parameters from the above mentioned paper
    def set_prior_hyperparameters( self, **kwargs ):
        #
        # ~~~ If any of the 3 hyper-parameters pi, sigma1, or sigma2 are unspecified, then use the class level defaults
        try:
            pi = kwargs["pi"]
        except KeyError:
            pi = self.default_pi
            my_warn(f'Hyper-parameter "pi" not specified. Using default value of {self.default_pi}.')
        try:
            sigma1 = kwargs["sigma1"]
        except KeyError:
            sigma1 = self.default_sigma1
            my_warn(f'Hyper-parameter "sigma1" not specified. Using default value of {self.default_sigma1}.')
        try:
            sigma2 = kwargs["sigma2"]
        except KeyError:
            sigma2 = self.default_sigma2
            my_warn(f'Hyper-parameter "sigma2" not specified. Using default value of {self.default_sigma2}.')
        #
        # ~~~ Check one or two features and then set the desired hyper-parameters as attributes of the class instance
        if not 0<pi<1 and sigma1>0 and sigma2>0:
            raise ValueError("The hyper-parameters of the mixture prior must be positive, and pi must be <1, as well, for specifying the a non-degenerate Gaussian mixture (equation (7) in https://arxiv.org/abs/1505.05424).")
        self.pi     = pi     if isinstance(pi,    torch.Tensor) else torch.tensor(pi)
        self.sigma1 = sigma1 if isinstance(sigma1,torch.Tensor) else torch.tensor(sigma1)
        self.sigma2 = sigma2 if isinstance(sigma2,torch.Tensor) else torch.tensor(sigma2)
    #
    # ~~~ Evaluate the log of the prior density (equation (7) in https://arxiv.org/abs/1505.05424)
    def estimate_expected_prior_log_density(self):
        #
        # ~~~ Gather the posterior parameters with repsect to which the expectation is computed
        mu_post     =  flatten_parameters(self.posterior_mean)
        sigma_post  =  flatten_parameters(self.posterior_std)
        z_sampled   =  flatten_parameters(self.realized_standard_posterior_sample)
        w_sampled   =  mu_post + sigma_post*z_sampled   # ~~~ w_sampled==F_\theta(z_sampled)
        #
        # ~~~ Compute the log_density of a Gaussian mixture (equation (7) in https://arxiv.org/abs/1505.05424)
        marginal_log_probs1  = -(w_sampled/self.sigma1)**2/2 - torch.log( math.sqrt(2*torch.pi)*self.sigma1 )
        marginal_log_probs2  = -(w_sampled/self.sigma2)**2/2 - torch.log( math.sqrt(2*torch.pi)*self.sigma2 )
        # marginal_log_density =  ( self.pi * marginal_log_probs1.exp() + (1-self.pi) * marginal_log_probs2.exp() ).log()
        # marginal_log_density = torch.where(
        #         torch.bitwise_or(
        #                 torch.isnan(marginal_log_density),
        #                 marginal_log_density.abs() == torch.inf
        #             ),
        #         torch.maximum(
        #                 self.pi.log() + marginal_log_probs1,
        #             (1-self.pi).log() + marginal_log_probs2
        #         ),
        #         marginal_log_density
        #     )
        # return marginal_log_density.sum()
        #
        # ~~~ If underflow/overflow, employ the approximation log( a*exp(x) + b*exp(y) ) \approx max( log(a)+x, log(b)+y ); viz. latter \leq former \leq \ln(2) + latter
        return torch.maximum(
                    self.pi.log() + marginal_log_probs1,
                (1-self.pi).log() + marginal_log_probs2
            ).sum()
    #
    # ~~~ Generate samples of a model with weights distributed according to the prior distribution (equation (7) in https://arxiv.org/abs/1505.05424)
    def prior_forward( self, x, n=1 ):
        #
        # ~~~ Stack n copies of x for bacthed multiplication with n different samples of the parameters (a loop would be simpler but less efficient)
        x = torch.stack(n*[x])
        #
        # ~~~ Basically, apply `x=layer(x)` for each layer in model, but resampling the weights and biases from linear layers
        for layer in self.posterior_mean:
            #
            # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
            if not isinstance( layer, nn.Linear ):
                x = layer(x)
            else:
                #
                # ~~~ Define a matrix full of samples from the Gaussian mixture prior (see https://stats.stackexchange.com/questions/70855/generating-random-variables-from-a-mixture-of-normal-distributions)
                u_weight =  torch.rand( n,*layer.weight.shape, dtype=x.dtype, device=x.device )
                z_weight = torch.randn( n,*layer.weight.shape, dtype=x.dtype, device=x.device )
                A = torch.where( u_weight<self.pi, self.sigma1*z_weight, self.sigma2*z_weight ) # ~~~ indices where u<pi are a sample from N(0,sigma1^2), and indices where u>pi are a sample from N(0,sigma2^2)
                x = torch.bmm( x, A.transpose(1,2) )                                             # ~~~ apply the appropriately distributed weights to this layer's input using batched matrix multiplication
                if layer.bias is not None:
                    u_bias =  torch.rand( n, 1, *layer.bias.shape, dtype=x.dtype, device=x.device )
                    z_bias = torch.randn( n, 1, *layer.bias.shape, dtype=x.dtype, device=x.device )
                    x += torch.where( u_bias<self.pi, self.sigma1*z_bias, self.sigma2*z_bias )  # ~~~ apply the appropriately distributed biases
        return x



### ~~~
## ~~~ Implement `estimate_expected_prior_log_density`, `prior_forward`, and `set_prior_hyperparameters` for the case in which the prior distribution is an independent location-scale family on weights (most commonly, Gaussian is used)
### ~~~

class ConventionalWeightPriorBNN(ConventionalVariationalFamilyBNN):
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
                posterior_standard_sampler,     # ~~~ should return a tensor of random samples from the reference distribution
                #
                # ~~~ Specify the location-scale family of the prior distribution
                prior_standard_log_density, # ~~~ too should be a callable that accepts generic torch.tensors as input but also works on numpy arrays, e.g. `lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) )` for Gaussian
                prior_standard_sampler,     # ~~~ too should return a tensor of random samples from the reference distribution
            ):
        super().__init__(
                *args,
                likelihood_std = likelihood_std,
                auto_projection = auto_projection,
                posterior_standard_log_density = posterior_standard_log_density,
                posterior_standard_sampler     = posterior_standard_sampler
            )
        #
        # ~~~ Define a prior on the weights
        with torch.no_grad():
            #
            # ~~~ First copy the architecture
            self.prior_mean = nonredundant_copy_of_module_list(self.posterior_mean)
            self.prior_std  = nonredundant_copy_of_module_list(self.posterior_mean)
            self.realized_standard_prior_sample = nonredundant_copy_of_module_list(self.posterior_mean)
            for p in self.realized_standard_prior_sample.parameters(): p.requires_grad = False
            #
            # ~~~ Don't train the prior
            for (mu,sigma) in zip( self.prior_mean.parameters(), self.prior_std.parameters() ):
                mu.requires_grad = False
                sigma.requires_grad = False
                mu.data = torch.zeros_like(mu.data) # ~~~ assign a prior mean of zero to the parameters
            #
            # ~~~ Set the formulas used for evaluating the log prior pdf and/or sampling from the prior distribution
            self.prior_log_density          = LocationScaleLogDensity(prior_standard_log_density)
            self.prior_standard_sampler     = prior_standard_sampler
            #
            # ~~~ Set the prior standard deviations
            self.default_prior_type = "torch.nn.init"   # ~~~ also supported are "Tom" and "IID"
            self.default_scale = torch.tensor(1.)
            self.set_prior_hyperparameters( prior_type=self.default_prior_type, scale=self.default_scale )
    #
    # ~~~ Allow these to be set at runtime
    def set_prior_hyperparameters( self, **kwargs ):
        #
        # ~~~ If any of the hyper-parameters are unspecified, then use the class level defaults
        try:
            prior_type = kwargs["prior_type"]
        except KeyError:
            prior_type = self.default_prior_type
            my_warn(f'Key word argument `prior_type` not specified; using default "{prior_type}". Available options are "torch.nn.init", "Tom", and "IID".')
        try:
            scale = kwargs["scale"]
        except KeyError:
            scale = self.default_scale
            my_warn(f'Key word argument `scale` not specified (should be positive, float); using default "{scale}".')
        #
        # ~~~ Check one or two features and then set the desired hyper-parameters as attributes of the class instance
        if not scale>0:
            raise ValueError(f'Variable `scale` should be a positive float.')
        if not prior_type in ( "torch.nn.init", "Tom", "IID" ):
            raise ValueError('Variable `prior_type` should be one of "torch.nn.init", "Tom", or "IID".')
        scale = scale if isinstance(scale,torch.Tensor) else torch.tensor(scale)
        #
        # ~~~ Implement prior_type=="torch.nn.init" (`scale` used later)
        if prior_type=="torch.nn.init": # ~~~ use the stanard deviation of the distribution of pytorch's default initialization
            for layer in self.prior_std:
                if isinstance(layer,nn.Linear):
                    std = std_per_layer(layer)
                    layer.weight.data = std * torch.ones_like(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data = std * torch.ones_like(layer.bias.data)
        #
        # ~~~ Implement prior_type=="Tom" (`scale` used later)
        if prior_type=="Tom":
            for p in self.prior_std.parameters():
                p.data = std_per_param(p)*torch.ones_like(p.data)
        #
        # ~~~ Implement prior_type=="IID" and use `scale`
        if prior_type=="IID":
            for p in self.prior_std.parameters():
                p.data = scale*torch.ones_like(p.data)
        else:
            #
            # ~~~ Scale the range of output, by scaling the parameters of the final linear layer, much like the scale paramter in a GP
            for layer in reversed(self.prior_std):
                if isinstance(layer,nn.Linear):
                    layer.weight.data *= scale
                    if layer.bias is not None:
                        layer.bias.data *= scale
                    break
    #
    # ~~~ Sample according to a "standard normal [or other] distribution in the shape of our neural network"
    def sample_from_standard_prior(self):
        with torch.no_grad():   # ~~~ theoretically the `no_grad()` context is redundant and unnecessary, but idk why not use it
            for p in self.realized_standard_prior_sample.parameters():
                p.data = self.prior_standard_sampler( *p.shape, device=p.device, dtype=p.dtype )
    #
    # ~~~ Allow a different standard distribution for the prior and posterior
    def resample_weights(self):
        self.sample_from_standard_posterior()
        self.sample_from_standard_prior()
    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def estimate_expected_prior_log_density(self):
        mu_post     =  flatten_parameters(self.posterior_mean)
        sigma_post  =  flatten_parameters(self.posterior_std)
        mu_prior    =  flatten_parameters(self.prior_mean)
        sigma_prior =  flatten_parameters(self.prior_std)
        z_sampled   =  flatten_parameters(self.realized_standard_prior_sample)
        w_sampled   =  mu_post + sigma_post*z_sampled   # ~~~ w_sampled==F_\theta(z_sampled)
        return self.prior_log_density( where=w_sampled, mu=mu_prior, sigma=sigma_prior )
    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just replace `posterior_mean` and `posterior_std` with `prior_mean` and `prior_std` in `forward`)
    def prior_forward( self, x, n=1 ):
        x = torch.stack(n*[x])  # ~~~ stack n copies of x for bacthed multiplication with n different samples of the parameters (a loop would be simpler but less efficient)
        for j, layer in enumerate(self.posterior_mean):
            #
            # ~~~ If this layer is just like relu or something, then there aren't any weights; just apply the layer and be done
            if not isinstance( layer, nn.Linear ):
                x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.prior_mean[j] # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer  =  self.prior_std[j] # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                z_sampled  =  self.prior_standard_sampler( n,*layer.weight.shape, dtype=x.dtype, device=x.device )
                A = mean_layer.weight + std_layer.weight * z_sampled
                x = torch.bmm( x, A.transpose(1,2) )
                if layer.bias is not None:
                    z_sampled = self.prior_standard_sampler( n,1,*layer.bias.shape, dtype=x.dtype, device=x.device )
                    b = mean_layer.bias + std_layer.bias * z_sampled
                    x += b
        return x



### ~~~
## ~~~ Define what most people are talking about when they say talk about BNN's
### ~~~

class ConventionalBNN(ConventionalWeightPriorBNN):
    def __init__(
                self,
                *args,
                likelihood_std = torch.tensor(0.001),
                auto_projection = True
            ):
        super().__init__(
                *args,
                likelihood_std  = likelihood_std,
                auto_projection = auto_projection,
                #
                # ~~~ For the variational family, use "fully factored Gaussian weights"
                posterior_standard_log_density = lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) ),
                posterior_standard_sampler     = torch.randn,
                #
                # ~~~ Use a "fully factored Gaussian prior" on weights
                prior_standard_log_density = lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) ),
                prior_standard_sampler     = torch.randn,
            )
        #
        # ~~~ Use the same "random seed" for both the prior and posterior ("this is an instance of a variance reduction technique known as common random numbers" source: https://arxiv.org/abs/1505.05424)
        for (z_post,z_prior) in zip( self.realized_standard_posterior_sample.parameters(), self.realized_standard_prior_sample.parameters() ):
            z_post.data = z_prior.data  # ~~~ updates to one are, also, reflected in the other after this
        self.sample_from_standard_posterior(counter_on=False)
    #
    # ~~~ Specify an exact formula for the KL divergence
    def compute_exact_weight_kl(self):
        mu_post = flatten_parameters(self.posterior_mean)
        sigma_post = flatten_parameters(self.posterior_std)
        mu_prior = flatten_parameters(self.prior_mean)
        sigma_prior = flatten_parameters(self.prior_std)
        return diagonal_gaussian_kl( mu_0=mu_post, sigma_0=sigma_post, mu_1=mu_prior, sigma_1=sigma_prior )
