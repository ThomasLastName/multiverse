
import math
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain     # ~~~ used to define the prior distribution on network weights

from bnns.utils import log_gaussian_pdf
from bnns.BayesianModule import BayesianModule

from quality_of_life.my_base_utils import my_warn
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list



### ~~~
## ~~~ A few helper routines not worth storing in `utils`
### ~~~

#
# ~~~ Propose a good "prior" standard deviation for a parameter group
def std_per_param(p):
    if len(p.shape)==2:
        #
        # ~~~ For weight matrices, use the standard deviation of pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
        gain = calculate_gain("relu")
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    elif len(p.shape)==1:
        #
        # ~~~ For bias vectors, just use variance==1/len(p) because `_calculate_fan_in_and_fan_out` throws a ValueError(""Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"")
        numb_pars = len(p)
        std = 1/math.sqrt(numb_pars)
    return torch.tensor( std, device=p.device, dtype=p.dtype )

#
# ~~~ Propose good a "prior" standard deviation for weights and biases of a linear layer; mimics pytorch's default initialization, but using a normal instead of uniform distribution (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
def std_per_layer(linear_layer):
    assert isinstance(linear_layer,nn.Linear)
    bound = 1 / math.sqrt(linear_layer.weight.size(1))  # ~~~ see the link above (https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2)
    std = bound / math.sqrt(3)  # ~~~ our reference distribution `uniform_(-bound,bound)` from the deafult pytorch weight initialization has standard deviation bound/sqrt(3), the value of which we copy
    return std

#
# ~~~ Flatten and concatenate all the parameters in a model
flatten_parameters = lambda model: torch.cat([ p.view(-1) for p in model.parameters() ])



### ~~~
## ~~~ Define a BNN with the necessary methods
### ~~~

#
# ~~~ Main class: intended to mimic nn.Sequential (STILL NO PRIOR DISTRIBUTION AT THIS LEVEL OF ABSTRACTION)
class IndependentLocationScaleBNN(BayesianModule):
    def __init__(
                self,
                *args,
                family_log_density,
                family_initializer,
                family_generator = None,
                conditional_std = torch.tensor(0.001)
            ):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        self.model_mean = nn.Sequential(*args)
        self.model_std  = nonredundant_copy_of_module_list( self.model_mean, sequential=True )
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
        self.family_log_density = family_log_density
        self.family_initializer = family_initializer # ~~~ this is "ducky" (https://en.wikipedia.org/wiki/Duck_typing); can be anything that modifies the input's `data` attribute in place
        self.family_generator   =   family_generator
        self.realized_standard_distribution = nonredundant_copy_of_module_list(self.model_mean)
        self.sample_from_standard_distribution()
        #
        # ~~~ Attributes determining the log likelihood density
        self.likelihood_model = "Gaussian"
        self.conditional_std = conditional_std
    # ~~~
    #
    ### ~~~
    ## ~~~ Basic methods such as "check that the weights are positive" and "make the weights positive" (`projection_step`)
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
                self.family_initializer(p)
    #
    # ~~~ Check that all the posterior standard deviations are positive
    def check_positive( self, forceful=False ):
        with torch.no_grad():
            if not flatten_parameters(self.model_std).min() > 0:
                if hasattr(self,"soft_projection"):
                    my_warn("`model_std` contains negative values.")
                if forceful:
                    self.hard_projection()
    #
    # ~~~ Whatever constraints we want the standard deviations to satisfy, define a projection onto the constraint such that hard_projection(sigma)==sigma if sigma already already satisfies the constraints
    @abstractmethod
    def hard_projection( self, p, tol=1e-6 ):
        raise NotImplementedError("The class IndependentLocationScaleBNN leaves the method `hard_projection` to be implented in user-defined subclasses, because it may depend on the prior distribution.")
    #
    # ~~~ Project the standard deviations to be positive
    def projection_step(self,soft):
        with torch.no_grad():
            for sigma in self.model_std.parameters():            
                sigma.data = self.soft_projection(sigma.data) if soft else self.hard_projection(sigma.data)
    #
    # ~~~ If not using projected gradient descent, then "parameterize the standard deviation pointwise" (as on page 4 of https://arxiv.org/pdf/1505.05424)
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
            raise ValueError(f' Unrecognized method="{method}". Currently, only method="Blundell" and "method=torchbnn" are supported.')
    #
    # ~~~ In Blundell et al. (https://arxiv.org/abs/1505.05424), the chain rule is implemented manually (this is necessary since pytorch doesn't allow in-place operations on the parameters to be included in the graph)
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
        self.check_positive()
        for j in range(self.n_layers):
            z = self.realized_standard_distribution[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            #
            # ~~~ If this layer is just like relu or something, then there aren't any weights; just apply the layer and be done
            if not isinstance( z, nn.modules.linear.Linear ):
                x = z(x)                            # ~~~ x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.model_mean[j]     # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer  =  self.model_std[j]     # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                A = mean_layer.weight + std_layer.weight * z.weight # ~~~ A = F_\theta(z.weight) is normal with the trainable (posterior) mean and std
                b = mean_layer.bias   +   std_layer.bias * z.bias   # ~~~ b = F_\theta(z.bias)   is normal with the trainable (posterior) mean and std
                x = x@A.T + b                                       # ~~~ apply the appropriately distributed weights to this layer's input
        return x
    #
    # ~~~ Compute ln( f_{Y \mid X,W}(F_\theta(z),x_train,y_train) ) at a point z sampled from the standard MVN distribution ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_likelihood_density( self, X, y ):
        #
        # ~~~ Record the last seen input, because it is recommended to include training data in the measurement set (see "Choosing the Measurement Set" in https://arxiv.org/abs/1903.05779)
        self.last_seen_x = X
        #
        # ~~~ The likelihood depends on task criterion: classification or regression
        if self.likelihood_model == "Gaussian":
            return log_gaussian_pdf( where=y, mu=self(X,resample_weights=False), sigma=self.conditional_std )  # ~~~ Y|X,W is assumed to be normal with mean self(X) and variance self.conditional_std (the latter being a tunable hyper-parameter)
        else:
            raise NotImplementedError("In the current version of the code, only the Gaussian likelihood (i.e., mean squared error) is implemented See issue ?????.")
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the loss in Bayes by Backprop
    ### ~~~
    #
    # ~~~ Compute \ln( q_\theta(F_\theta(z)) ) at a point z sampled from the standard MVN distribution, where q_\theta is the posterior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_posterior_density(self):
        mu_post = flatten_parameters(self.model_mean)
        sigma_post = flatten_parameters(self.model_std)
        z_sampled = flatten_parameters(self.realized_standard_distribution)
        w_sampled = mu_post + sigma_post*z_sampled
        return self.family_log_density( where=w_sampled, mu=mu_post, sigma=sigma_post )
    # def sample_new_measurement_set(self,n=200):
    #     #
    #     # ~~~ In the common case that the inputs are standardized, then standard random normal vectors are "points like our model's inputs"
    #     device, dtype = self.infer_device_and_dtype()
    #     self.measurement_set = torch.randn( size=(n,self.in_features), device=device, dtype=dtype )