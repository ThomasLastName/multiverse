
import torch
from torch import nn

from bnns.utils import flatten_parameters, log_gaussian_pdf, diagonal_gaussian_kl, std_per_param, std_per_layer
from bnns.NoPriorBNNs import ConventionalSequentialBNN

from quality_of_life.my_base_utils  import my_warn
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list

#
# ~~~ Implement `estimate_expected_log_prior` and `prior_forward` for the "homoskedastic" mixture prior on the network weights employed in Blundell et al. 2015 (https://arxiv.org/abs/1505.05424)
class MixtureWeightPrior2015BNN(ConventionalSequentialBNN):
    def __init__(
                self,
                *args,
                conditional_std = torch.tensor(0.001),
                auto_projection = True,
                model_log_density = log_gaussian_pdf
            ):
        super().__init__(
                *args,
                conditional_std = conditional_std,
                auto_projection = auto_projection,
                model_log_density = model_log_density,
                model_initializer = nn.init.normal_ # ~~~ we'll use *normal* distribution for sampling from the *gaussian* mixture prior
            )
        #
        # ~~~ Set default values for hyper-parameters of the prior found here: https://github.com/danielkelshaw/WeightUncertainty/blob/master/torchwu/bayes_linear.py
        self.default_pi = torch.tensor(0.5) # ~~~ WARNING: this is not the mathematical constant pi\approx3.14. I have beef with Blundell et al.'s use of "\pi" to refer to a value between 0 and 1...
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
    def estimate_expected_log_prior(self):
        #
        # ~~~ Gather the posterior parameters with repsect to which the expectation is computed
        mu_post     =  flatten_parameters(self.model_mean)
        sigma_post  =  flatten_parameters(self.model_std)
        z_sampled   =  flatten_parameters(self.realized_standard_distribution)
        w_sampled   =  mu_post + sigma_post*z_sampled   # ~~~ w_sampled==F_\theta(z_sampled)
        #
        # ~~~ Note, a few flops could be saved by re-using the intermediate values defined in log_gaussian_pdf, NBD
        log_density1 = log_gaussian_pdf( where=w_sampled, mu=torch.zeros_like(w_sampled), sigma=self.sigma1 )
        log_density2 = log_gaussian_pdf( where=w_sampled, mu=torch.zeros_like(w_sampled), sigma=self.sigma2 )
        return ( self.pi*log_density1.exp() + (1-self.pi)*log_density2.exp() ).log()
    #
    # ~~~ Generate samples of a model with weights distributed according to the prior distribution (equation (7) in https://arxiv.org/abs/1505.05424)
    def prior_forward( self, x, n=1 ):
        if n==1:
            #
            # ~~~ Basically, apply `x=layer(x)` for each layer in model, but with a twist for linear layers
            self.sample_from_standard_distribution()        # ~~~ this method re-generates the values of weights and biases in `self.realized_standard_distribution` (IID standard normal)
            for j in range(self.n_layers):
                z = self.realized_standard_distribution[j]  # ~~~ the network's j'th layer, but with IID standard normal weights and biases
                #
                # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
                if not isinstance( z, nn.Linear ):
                    x = z(x)                                # ~~~ x = layer(x)
                #
                # ~~~ Aforementioned twist is that we fill the weights and biases with samples from the prior distribution
                else:
                    #
                    # ~~~ To build samples from a Gaussian mixture, we first sample from U[0,1] (see https://stats.stackexchange.com/questions/70855/generating-random-variables-from-a-mixture-of-normal-distributions)
                    u_weight = torch.rand_like(z.weight)
                    u_bias   = torch.rand_like(z.bias)
                    #
                    # ~~~ Define A and b which are samples from the Gaussian mixture prior (see https://stats.stackexchange.com/questions/70855/generating-random-variables-from-a-mixture-of-normal-distributions)
                    A = torch.where( u_weight<self.pi, self.sigma1*z.weight, self.sigma2*z.weight ) # ~~~ indices where u<pi are a sample from N(0,sigma1^2), and...
                    b = torch.where( u_bias<self.pi,   self.sigma1*z.bias,   self.sigma2*z.bias   ) # ~~~ indices where u>pi are a sample from N(0,sigma2^2)
                    x = x@A.T + b  # ~~~ apply the appropriately distributed weights to this layer's input
            return x
        else:
            return torch.row_stack([ self.prior_forward(x,n=1).flatten() for _ in range(n) ])

#
# ~~ ~ Implement `estimate_expected_log_prior` and `prior_forward` for a "fully factored" Gaussian prior on the network weights


#
class ConventionalWeightPriorSequentialBNN(ConventionalSequentialBNN):
    def __init__(
                self,
                *args,
                model_log_density,
                model_initializer,
                prior_log_density,
                prior_initializer,
                conditional_std = torch.tensor(0.001),
                auto_projection = True
            ):
        super().__init__(
                *args,
                model_log_density = model_log_density,
                model_initializer = model_initializer,
                conditional_std = conditional_std,
                auto_projection = auto_projection
            )
        #
        # ~~~ Define a prior on the weights
        with torch.no_grad():
            #
            # ~~~ First copy the architecture
            self.prior_mean = nonredundant_copy_of_module_list(self.model_mean)
            self.prior_std  = nonredundant_copy_of_module_list(self.model_mean)
            #
            # ~~~ Don't train the prior
            for (mu,sigma) in zip( self.prior_mean.parameters(), self.prior_std.parameters() ):
                mu.requires_grad = False
                sigma.requires_grad = False
                mu.data = torch.zeros_like(mu.data) # ~~~ assign a prior mean of zero to the parameters
            #
            # ~~~ Set the prior standard deviation
            self.default_prior_type = "torch.nn.init"   # ~~~ also supported are "Tom" and "IID"
            self.default_prior_scale = torch.tensor(1.)
            self.set_prior_hyperparameters( prior_type=self.default_prior_type, scale=self.default_prior_scale )
            self.prior_log_density = prior_log_density
            self.prior_initializer = prior_initializer
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
            prior_scale = kwargs["prior_scale"]
        except KeyError:
            prior_scale = self.prior_scale
            my_warn(f'Key word argument `prior_scale` not specified (should be positive, float); using default "{prior_scale}".')
        #
        # ~~~ Check one or two features and then set the desired hyper-parameters as attributes of the class instance
        if not prior_scale>0:
            raise ValueError(f'Variable `prior_scale` should be a positive float.')
        prior_scale = prior_scale if isinstance(prior_scale,torch.Tensor) else torch.tensor(prior_scale)
        #
        # ~~~ Implement prior_type=="torch.nn.init" (`prior_scale` used later)
        if prior_type=="torch.nn.init": # ~~~ use the stanard deviation of the distribution of pytorch's default initialization
            for layer in self.prior_std:
                if isinstance(layer,nn.Linear):
                    std = std_per_layer(layer)
                    layer.weight.data = std * torch.ones_like(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data = std * torch.ones_like(layer.bias.data)
        #
        # ~~~ Implement prior_type=="Tom" (`prior_scale` used later)
        if prior_type=="Tom":
            for p in self.prior_std.parameters():
                p.data = std_per_param(p)*torch.ones_like(p.data)
        #
        # ~~~ Implement prior_type=="IID" and use `prior_scale`
        if prior_type=="IID":
            for p in self.prior_std.parameters():
                p.data = prior_scale*torch.ones_like(p.data)
        else:
            #
            # ~~~ Scale the range of output, by scaling the parameters of the final linear layer, much like the scale paramter in a GP
            for layer in reversed(self.prior_std):
                if isinstance(layer,nn.Linear):
                    layer.weight.data *= prior_scale
                    if layer.bias is not None:
                        layer.bias.data *= prior_scale
                    break
    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def estimate_expected_log_prior(self):
        mu_post     =  flatten_parameters(self.model_mean)
        sigma_post  =  flatten_parameters(self.model_std)
        mu_prior    =  flatten_parameters(self.prior_mean)
        sigma_prior =  flatten_parameters(self.prior_std)
        z_sampled   =  flatten_parameters(self.realized_standard_distribution)
        w_sampled   =  mu_post + sigma_post*z_sampled   # ~~~ w_sampled==F_\theta(z_sampled)
        return log_gaussian_pdf( where=w_sampled, mu=mu_prior, sigma=sigma_prior )
    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just replace `model_mean` and `model_std` with `prior_mean` and `prior_std` in `forward`)
    def prior_forward( self, x, n=1 ):
        if n==1:
            #
            # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
            self.sample_from_standard_distribution()          # ~~~ this method re-generates the values of weights and biases in `self.realized_standard_distribution` (IID standard normal)
            for j in range(self.n_layers):
                z = self.realized_standard_distribution[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
                #
                # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
                if not isinstance( z, nn.Linear ):
                    x = z(x)                            # ~~~ x = layer(x)
                #
                # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
                else:
                    mean_layer = self.prior_mean[j]     # ~~~ the user-specified prior means of this layer's parameters
                    std_layer  =  self.prior_std[j]     # ~~~ the user-specified prior standard deviations of this layer's parameters
                    A = mean_layer.weight + std_layer.weight * z.weight # ~~~ A = F_\theta(z.weight) is normal with the user-specified prior mean and std
                    b = mean_layer.bias   +   std_layer.bias * z.bias   # ~~~ b = F_\theta(z.bias)   is normal with the user-specified prior mean and std
                    x = x@A.T + b                       # ~~~ apply the appropriately distributed weights to this layer's input
            return x
        else:
            return torch.row_stack([ self.prior_forward(x,n=1).flatten() for _ in range(n) ])

#
# ~~~ Define what most people are talking about when they say talk about BNN's
class SequentialGaussianBNN(ConventionalWeightPriorSequentialBNN):
    def __init__(
                self,
                *args,
                conditional_std = torch.tensor(0.001),
                auto_projection = True
            ):
        super().__init__(
                *args,
                model_log_density = log_gaussian_pdf,
                model_initializer = nn.init.normal_,
                prior_log_density = log_gaussian_pdf,
                prior_initializer = nn.init.normal_,
                conditional_std = conditional_std,
                auto_projection = auto_projection
            )
    #
    # ~~~ Define alias for partial backwards compatibility
    def sample_from_standard_normal(self):
        self.sample_from_standard_distribution()
    #
    # ~~~ Specify an exact formula for the KL divergence
    def exact_weight_kl(self):
        mu_post = flatten_parameters(self.model_mean)
        sigma_post = flatten_parameters(self.model_std)
        mu_prior = flatten_parameters(self.prior_mean)
        sigma_prior = flatten_parameters(self.prior_std)
        return diagonal_gaussian_kl( mu_0=mu_post, sigma_0=sigma_post, mu_1=mu_prior, sigma_1=sigma_prior )
