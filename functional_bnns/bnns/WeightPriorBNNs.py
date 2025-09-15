import math
from tqdm import tqdm
import torch
from torch import nn

from bnns.utils.math import (
    diagonal_gaussian_kl,
    std_per_param,
    std_per_layer,
    LocationScaleLogDensity,
)
from bnns.utils.handling import (
    flatten_parameters,
    support_for_progress_bars,
    nonredundant_copy_of_module_list,
)
from bnns.NoPriorBNNs import FullSupportIndepLocScaleBNN


### ~~~
## ~~~ Implement `estimate_expected_prior_log_density`, `prior_forward`, and `set_prior_hyperparameters` for the "homoskedastic" mixture prior on the network weights employed in Blundell et al. 2015 (https://arxiv.org/abs/1505.05424)
### ~~~


class MixturePrior2015BNN(FullSupportIndepLocScaleBNN):
    def __init__(
        self,
        *args,
        prior_generator=None,  # ~~~ the only new kwarg that this sub-class introduces
        pi=0.5,
        sigma1=1,
        sigma2=0.002,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.set_prior_hyperparameters(pi=pi, sigma1=sigma1, sigma2=sigma2)
        self.prior_generator = prior_generator

    #
    # ~~~ Allow the hyper-parameters of the prior distribution to be set at runtime
    def set_prior_hyperparameters(self, pi, sigma1, sigma2):
        if not 0 < pi < 1 and sigma1 > 0 and sigma2 > 0:
            raise ValueError(
                "For the Gaussian mixture (equation (7) in https://arxiv.org/abs/1505.05424), we expect sigma1,sigma2,pi>0 and pi<1, but found pi={pi}, sigma1={sigma1}, and sigma2={sigma2} )."
            )
        self.pi = nn.Parameter(
            pi if isinstance(pi, torch.Tensor) else torch.tensor(pi),
            requires_grad=False,
        )
        self.sigma1 = nn.Parameter(
            sigma1 if isinstance(sigma1, torch.Tensor) else torch.tensor(sigma1),
            requires_grad=False,
        )
        self.sigma2 = nn.Parameter(
            sigma2 if isinstance(sigma2, torch.Tensor) else torch.tensor(sigma2),
            requires_grad=False,
        )

    #
    # ~~~ Evaluate the log of the prior density (equation (7) in https://arxiv.org/abs/1505.05424) at a point sampled from the variational distribution
    def estimate_expected_prior_log_density(self):
        #
        # ~~~ Gather the posterior parameters with repsect to which the expectation is computed
        mu_post = flatten_parameters(self.posterior_mean)
        sigma_post = flatten_parameters(self.posterior_std)
        z_sampled = flatten_parameters(self.realized_standard_posterior_sample)
        w_sampled = (
            mu_post + sigma_post * z_sampled
        )  # ~~~ w_sampled == F_\theta(z_sampled) is a sample from the variational distribution
        #
        # ~~~ Compute the log_density of a Gaussian mixture (equation (7) in https://arxiv.org/abs/1505.05424)
        marginal_log_probs1 = -((w_sampled / self.sigma1) ** 2) / 2 - torch.log(
            math.sqrt(2 * torch.pi) * self.sigma1
        )
        marginal_log_probs2 = -((w_sampled / self.sigma2) ** 2) / 2 - torch.log(
            math.sqrt(2 * torch.pi) * self.sigma2
        )
        #
        # ~~~ Why doesn't the following commented out code work?
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
            (1 - self.pi).log() + marginal_log_probs2,
        ).sum()

    #
    # ~~~ Generate samples of a model with weights distributed according to the prior distribution (equation (7) in https://arxiv.org/abs/1505.05424)
    def prior_forward(self, x, n=1):
        #
        # ~~~ Stack n copies of x for bacthed multiplication with n different samples of the parameters (a loop would be simpler but less efficient)
        x = torch.stack(n * [x])
        #
        # ~~~ Basically, apply `x=layer(x)` for each layer in model, but resampling the weights and biases from linear layers
        for layer in self.posterior_mean:
            #
            # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
            if not isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                #
                # ~~~ Define a matrix full of samples from the Gaussian mixture prior (see https://stats.stackexchange.com/questions/70855/generating-random-variables-from-a-mixture-of-normal-distributions)
                u_weight = torch.rand(
                    n,
                    *layer.weight.shape,
                    generator=self.prior_generator,
                    dtype=x.dtype,
                    device=x.device,
                )
                z_weight = torch.randn(
                    n,
                    *layer.weight.shape,
                    generator=self.prior_generator,
                    dtype=x.dtype,
                    device=x.device,
                )
                A = torch.where(
                    u_weight < self.pi, self.sigma1 * z_weight, self.sigma2 * z_weight
                )  # ~~~ indices where u<pi are a sample from N(0,sigma1^2), and indices where u>pi are a sample from N(0,sigma2^2)
                x = torch.bmm(
                    x, A.transpose(1, 2)
                )  # ~~~ apply the appropriately distributed weights to this layer's input using batched matrix multiplication
                if layer.bias is not None:
                    u_bias = torch.rand(
                        n,
                        1,
                        *layer.bias.shape,
                        generator=self.prior_generator,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    z_bias = torch.randn(
                        n,
                        1,
                        *layer.bias.shape,
                        generator=self.prior_generator,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    x += torch.where(
                        u_bias < self.pi, self.sigma1 * z_bias, self.sigma2 * z_bias
                    )  # ~~~ apply the appropriately distributed biases
        return x

    #
    # ~~~ Infer good prior hyperparameters by maximizing the log posterior
    def MLE_for_prior_hyperparameters(
        self, dataloader, likelihood_too=True, n_iter=500, projection_tol=1e-6, n_pi=21
    ):
        #
        # ~~~ Ready the variables to be optimized
        self.sigma1.requires_grad = True
        self.sigma2.requires_grad = True
        if likelihood_too:
            self.likelihood_std.requires_grad = True
        PI = torch.linspace(
            0, 1, n_pi
        )  # ~~~ prior_forward is not a differentiable function of the hyper-parameter pi, so we will use a grid search for pi only
        #
        # ~~~ Optimize the log posterior
        with support_for_progress_bars():
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
            pbar = tqdm(
                desc="Tuning Prior Hyper-Parameters",
                total=n_iter,
                initial=0,
                ascii=" >=",
            )
            self.watch_the_count = False
            while pbar.n < n_iter:
                for X, y in dataloader:
                    losses = []
                    for pi in PI:
                        self.pi.data = pi
                        losses.append(
                            -self.estimate_expected_log_likelihood(X, y, prior=True)
                        )
                    losses = torch.stack(losses)
                    loss = losses.max()
                    loss.backward()
                    optimizer.step()
                    self.pi.data = PI[losses.argmax().item()]
                    self.resample_weights()
                    #
                    # ~~~ Project onto the constraint set
                    with torch.no_grad():
                        self.likelihood_std.data.clamp_(min=projection_tol)
                        self.sigma1.data.clamp_(min=projection_tol)
                        self.sigma2.data.clamp_(min=projection_tol)
                        info = {
                            "log-lik": f"{loss.item():<4.4f}",
                            "pi": f"{self.pi.item():<4.4f}",
                            "s1": f"{self.sigma1.item():<4.4f}",
                            "s2": f"{self.sigma2.item():<4.4f}",
                        }
                        if likelihood_too:
                            info["noise_std"] = (f"{self.likelihood_std.item():<4.4f}",)
                        pbar.set_postfix(info)
                        pbar.update()
                        if pbar.n >= n_iter:
                            break
        self.sigma1.requires_grad = False
        self.sigma2.requires_grad = False
        if likelihood_too:
            self.likelihood_std.requires_grad = False
        self.watch_the_count = True


### ~~~
## ~~~ Implement `estimate_expected_prior_log_density`, `prior_forward`, and `set_prior_hyperparameters` for the case in which the prior distribution is an independent location-scale family on weights (most commonly, Gaussian is used)
### ~~~


class IndepLocScalePriorBNN(FullSupportIndepLocScaleBNN):
    def __init__(
        #
        # ~~~ Architecture and stuff
        self,
        *args,
        #
        # ~~~ Specify the location-scale family of the prior distribution
        prior_distribution=torch.distributions.Normal,  # ~~~ either, specify this, of specify the following two methods
        prior_standard_log_density=None,  # ~~~ should be a callable that accepts generic torch.tensors as input, but ideally also works on numpy arrays (otherwise `check_moments` will fail), e.g. `lambda z: -z**2/2 - math.log( math.sqrt(2*torch.pi) )` for Gaussian
        prior_standard_sampler=None,  # ~~~ should be a callable that returns a tensor of random samples from the distribution with mean 0 and variance 1, e.g., `torch.randn` for Gaussian
        #
        # ~~~ Specify the spread of the location-scale prior
        prior_type="torch.nn.init",  # ~~~ also accepted are "Xavier" and "IID"
        scale=1.0,
        gain_multiplier=1.0,
        #
        # ~~~ The other kwargs used by parent classes
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        #
        # ~~~ First, copy the architecture (this is kind of like metadata used to define the prior)
        self.prior_mean = nonredundant_copy_of_module_list(self.posterior_mean)
        self.prior_std = nonredundant_copy_of_module_list(self.posterior_mean)
        #
        # ~~~ Don't train the prior; also, use mean zero weights
        for mu, sigma in zip(self.prior_mean.parameters(), self.prior_std.parameters()):
            mu.requires_grad = False
            sigma.requires_grad = False
            with torch.no_grad():
                mu.data = torch.zeros_like(
                    mu.data
                )  # ~~~ assign a prior mean of zero to the parameters
        #
        # ~~~ Define information about the location scale family of the prior distribution
        try:
            check_moments = kwargs["check_moments"]
        except KeyError:
            check_moments = True
        if (prior_standard_log_density is None) ^ (
            prior_standard_sampler is None
        ):  # ~~~ one is specified, but not both are
            raise ValueError(
                "The arguments `posterior_standard_log_density` and `posterior_standard_sampler` should either both be specified, or both be `None`."
            )
        if (prior_standard_log_density is None) and (
            prior_standard_sampler is None
        ):  # ~~~ if neither are specified, then use `posterior_distribution` to specify them
            if not issubclass(prior_distribution, torch.distributions.Distribution):
                raise ValueError(
                    "The posterior distribution must be a subclass of torch.distributions.Distribution"
                )
            prior_standard_distribution = prior_distribution(self.zero, self.one)
            mean, std = (
                prior_standard_distribution.mean.item(),
                prior_standard_distribution.stddev.item(),
            )
            prior_standard_sampler = (
                lambda *args, **kwargs: (
                    prior_standard_distribution.sample(args) - mean
                )
                / std
            )  # ~~~ at the time of writing, this does not accep
            prior_standard_log_density = lambda z: prior_standard_distribution.log_prob(
                mean + std * z
            ) + math.log(std)
            check_moments = False
        self.prior_log_density = LocationScaleLogDensity(
            prior_standard_log_density, check_moments=check_moments
        )
        self.prior_standard_sampler = prior_standard_sampler
        self.set_prior_hyperparameters(
            prior_type=prior_type, scale=scale, gain_multiplier=gain_multiplier
        )

    #
    # ~~~ Allow these to be set at runtime
    def set_prior_hyperparameters(self, prior_type, scale, gain_multiplier):
        #
        # ~~~ Check one or two features and then set the desired hyper-parameters as attributes of the class instance
        if not scale > 0:
            raise ValueError(
                f"Variable `scale` should be a positive float, not {scale}."
            )
        if not prior_type in ("torch.nn.init", "Xavier", "IID"):
            raise ValueError(
                f'Variable `prior_type` should be one of "torch.nn.init", "Xavier", or "IID", not {prior_type}.'
            )
        #
        # ~~~ Implement prior_type=="torch.nn.init"
        if (
            prior_type == "torch.nn.init"
        ):  # ~~~ use the stanard deviation of the distribution of pytorch's default initialization
            for layer in self.prior_std:
                if isinstance(layer, nn.Linear):
                    std = gain_multiplier * std_per_layer(layer)
                    layer.weight.data = std * torch.ones_like(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data = std * torch.ones_like(layer.bias.data)
        #
        # ~~~ Implement prior_type=="Xavier"
        if prior_type == "Xavier":
            for p in self.prior_std.parameters():
                p.data = gain_multiplier * std_per_param(p) * torch.ones_like(p.data)
        #
        # ~~~ Implement prior_type=="IID"
        if prior_type == "IID":
            for p in self.prior_std.parameters():
                p.data = gain_multiplier * torch.ones_like(p.data)
        #
        # ~~~ Scale the range of output, by scaling the parameters of the final linear layer, much like the scale paramter in a GP
        for layer in reversed(self.prior_std):
            if isinstance(layer, nn.Linear):
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data *= scale
                break

    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def estimate_expected_prior_log_density(self):
        mu_post = flatten_parameters(self.posterior_mean)
        sigma_post = flatten_parameters(self.posterior_std)
        mu_prior = flatten_parameters(self.prior_mean)
        sigma_prior = flatten_parameters(self.prior_std)
        z_sampled = flatten_parameters(self.realized_standard_posterior_sample)
        w_sampled = (
            mu_post + sigma_post * z_sampled
        )  # ~~~ w_sampled==F_\theta(z_sampled)
        return self.prior_log_density(where=w_sampled, mu=mu_prior, sigma=sigma_prior)

    #
    # ~~~ Define how to sample from the priorly distributed outputs of the network (just replace `posterior_mean` and `posterior_std` with `prior_mean` and `prior_std` in `forward`)
    def prior_forward(self, x, n=1):
        x = torch.stack(
            n * [x]
        )  # ~~~ stack n copies of x for bacthed multiplication with n different samples of the parameters (a loop would be simpler but less efficient)
        for j, layer in enumerate(self.posterior_mean):
            #
            # ~~~ If this layer is just like relu or something, then there aren't any weights; just apply the layer and be done
            if not isinstance(layer, nn.Linear):
                x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.prior_mean[
                    j
                ]  # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer = self.prior_std[
                    j
                ]  # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                z_sampled = self.prior_standard_sampler(
                    n, *layer.weight.shape, dtype=x.dtype, device=x.device
                )
                A = mean_layer.weight + std_layer.weight * z_sampled
                x = torch.bmm(x, A.transpose(1, 2))
                if layer.bias is not None:
                    z_sampled = self.prior_standard_sampler(
                        n, 1, *layer.bias.shape, dtype=x.dtype, device=x.device
                    )
                    b = mean_layer.bias + std_layer.bias * z_sampled
                    x += b
        return x


### ~~~
## ~~~ Define what most people are talking about when they say "Bayesian neural networks"
### ~~~


class GaussianBNN(IndepLocScalePriorBNN):
    def __init__(
        self,
        *args,
        likelihood_std=torch.tensor(0.01),
        posterior_generator=None,
        prior_generator=None,
        posterior_distribution=None,  # ~~~ un-used argument for API compatibility
        **kwargs,
    ):
        if posterior_distribution is torch.distributions.Normal:
            posterior_distribution = None
        assert (
            posterior_distribution is None
        ), f"GaussianBNN simply implements a Gaussian distribution, which conflicts with the supplied value of the `posterior_distribution` keyword argument: {posterior_distribution}. Please specify the `posterior_distribution` keyword argument to `None`."
        super().__init__(
            *args,
            likelihood_std=likelihood_std,
            posterior_standard_log_density=lambda z: -(z**2) / 2
            - math.log(math.sqrt(2 * torch.pi)),
            posterior_standard_sampler=lambda *shape, **kwargs: torch.randn(
                *shape, generator=posterior_generator, **kwargs
            ),
            prior_standard_log_density=lambda z: -(z**2) / 2
            - math.log(math.sqrt(2 * torch.pi)),
            prior_standard_sampler=lambda *shape, **kwargs: torch.randn(
                *shape, generator=prior_generator, **kwargs
            ),
            check_moments = False,
            **kwargs,
        )

    #
    # ~~~ Specify an exact formula for the KL divergence
    def compute_exact_weight_kl(self):
        mu_post = flatten_parameters(self.posterior_mean)
        sigma_post = flatten_parameters(self.posterior_std)
        mu_prior = flatten_parameters(self.prior_mean)
        sigma_prior = flatten_parameters(self.prior_std)
        return diagonal_gaussian_kl(
            mu_0=mu_post, sigma_0=sigma_post, mu_1=mu_prior, sigma_1=sigma_prior
        )
