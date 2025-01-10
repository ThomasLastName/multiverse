
import torch
from torch import nn

from bnns.utils import log_gaussian_pdf, diagonal_gaussian_kl
from bnns.IndependentLocationScaleBNN import IndependentLocationScaleBNN, std_per_param, std_per_layer

from quality_of_life.my_base_utils import my_warn
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list

#
# ~~~ Implement `log_prior_density` and `prior_forward` for a "fully factored" Gaussian prior on the network weights
class GaussianWeightPrior(IndependentLocationScaleBNN):
    def __init__(
                self,
                *args,
                family_log_density,
                family_initializer,
                conditional_std = torch.tensor(0.001),
                comparable_to_default_torch_init = False,
                prior_scale = 1.0,
                prior_labels = None
            ):
        super().__init__(
                *args,
                family_log_density = family_log_density,
                family_initializer = family_initializer,
                conditional_std = conditional_std
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
            # ~~~ If labels are provided, use them to set the mean of the final bias
            if prior_labels is not None:
                mean_of_response = prior_labels.mean(dim=0)
                mu.data += mean_of_response
            #
            # ~~~ Set the prior standard deviation
            if comparable_to_default_torch_init:
                for layer in self.prior_std:
                    if isinstance(layer,nn.Linear):
                        std = std_per_layer(layer)
                        layer.weight.data = std * torch.ones_like(layer.weight.data)
                        if layer.bias is not None:
                            layer.bias.data = std * torch.ones_like(layer.bias.data)
            else:
                for p in self.prior_std.parameters():
                    p.data = std_per_param(p)*torch.ones_like(p.data)
            #
            # ~~~ Scale the range of output, much like the scale paramter in a GP
            if isinstance( self.prior_std[-1], nn.Linear ):
                for p in self.prior_std[-1].parameters():
                    p.data *= prior_scale
            elif not prior_scale==1:
                my_warn(f"`scale` assumes the final layer is `nn.Linear`, but found instead {type(self.prior_std[-1])}. The supplied `scale={scale}` was ignored.")
    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_prior_density(self):
        log_prior = 0.
        for ( mu_post, sigma_post, mu_prior, sigma_prior, z_sampled ) in zip(
                    self.model_mean.parameters(),
                    self.model_std.parameters(),
                    self.prior_mean.parameters(),
                    self.prior_std.parameters(),
                    self.realized_standard_distribution.parameters()
                ):
            w_sampled = mu_post + sigma_post*z_sampled  # ~~~ w_sampled==F_\theta(z_sampled)
            log_prior += log_gaussian_pdf( where=w_sampled, mu=mu_prior, sigma=sigma_prior )
        return log_prior
    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just replace `model_mean` and `model_std` with `prior_mean` and `prior_std` in `forward`)
    def prior_forward( self, x, n=1 ):
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        self.sample_from_standard_distribution()          # ~~~ this method re-generates the values of weights and biases in `self.realized_standard_distribution` (IID standard normal)
        for j in range(self.n_layers):
            z = self.realized_standard_distribution[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            #
            # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
            if not isinstance( z, nn.modules.linear.Linear ):
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

#
# ~~~ Define what most people are talking about when they say talk about BNN's
class SequentialGaussianBNN(GaussianWeightPrior):
    def __init__(
                self,
                *args,
                conditional_std = torch.tensor(0.001)
            ):
        super().__init__(
                *args,
                family_log_density = log_gaussian_pdf,
                family_initializer = nn.init.normal_,
                conditional_std = conditional_std
            )
    #
    # ~~~ Define alias for partial backwards compatibility
    def sample_from_standard_normal(self):
        self.sample_from_standard_distribution()
    #
    # ~~~ Specify an exact formula for the KL divergence
    def exact_weight_kl(self):
        kl_div = 0.
        #
        # ~~~ Because the weights and biases are mutually independent, the entropy is *additive* like log-density (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties)
        for ( mu_post, sigma_post, mu_prior, sigma_prior ) in zip(
                    self.model_mean.parameters(),
                    self.model_std.parameters(),
                    self.prior_mean.parameters(),
                    self.prior_std.parameters()
                ):
            kl_div += diagonal_gaussian_kl( mu_0=mu_post, sigma_0=sigma_post, mu_1=mu_prior, sigma_1=sigma_prior )
        return kl_div

#
# ~~~ Implement `log_prior_density` and `prior_forward` for the "homoskedastic" mixture prior on the network weights employed in Blundell et al. 2015 (https://arxiv.org/abs/1505.05424)
class MixtureWeightPrior2015(IndependentLocationScaleBNN):
    def __init__(
            self,
            *args,
            conditional_std = torch.tensor(0.001),
            comparable_to_default_torch_init = False,
            prior_scale = 1.0,
            prior_labels = None
        ):
        super().__init__( *args, conditional_std=conditional_std )
