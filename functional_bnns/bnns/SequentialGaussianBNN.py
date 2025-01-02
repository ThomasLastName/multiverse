
import math
import torch
from torch import nn
from torch.func import jacrev, functional_call
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain     # ~~~ used to define the prior distribution on network weights
from bnns.SSGE import SpectralSteinEstimator as SSGE
from bnns.utils import log_gaussian_pdf, diagonal_gaussian_kl, manual_Jacobian
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
    return std

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
# ~~~ Main class: intended to mimic nn.Sequential
class SequentialGaussianBNN(nn.Module):
    def __init__(self,*args):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        self.model_mean = nn.Sequential(*args)
        self.model_std  = nonredundant_copy_of_module_list( self.model_mean, sequential=True )
        #
        # ~~~ Basic information about the model: in_features, out_features, and n_layers, etc.
        self.n_layers   = len(self.model_mean)
        self.out_features = self.model_mean[-1].out_features
        for layer in self.model_mean:
            if hasattr(layer,"in_features"):    # ~~~ the first layer with an `in_features` attribute
                self.in_features = layer.in_features 
                break
        #
        # ~~~ Define a prior on the weights
        with torch.no_grad():
            #
            # ~~~ Define the prior means: first copy the architecture, then set requires_grad=False and assign the desired mean values (==zero, for now)
            self.prior_mean = nonredundant_copy_of_module_list(self.model_mean)
            for p in self.prior_mean.parameters():
                p.requires_grad = False             # ~~~ don't train the prior
                p.data = torch.zeros_like(p.data)   # ~~~ assign the desired prior mean values
            #
            # ~~~ Define the prior std. dev.'s: first copy the architecture, then set requires_grad=False, and finally assign the desired std values in a way that mimics pytorch's default initialization
            self.prior_std = nonredundant_copy_of_module_list(self.model_mean)  # ~~~ copy the architecture
            for p in self.prior_std.parameters():
                p.requires_grad = False                     # ~~~ don't train the prior
                p.data = std_per_param(p)*torch.ones_like(p.data) # ~~~ assign the desired prior standard deviation values
        #
        # ~~~ Define a "standard normal distribution in the shape of our neural network"
        self.realized_standard_normal = nonredundant_copy_of_module_list(self.model_mean)
        for p in self.realized_standard_normal.parameters():
            p.requires_grad = False
            nn.init.normal_(p)
        #
        # ~~~ Define the assumed level of noise in the training data: when this is set to smaller values, the model "pays more attention" to the data, and fits it more aggresively (can also be a vector)
        self.conditional_std = torch.tensor(0.001)
        #
        # ~~~ Opt to project onto [projection_tol,Inf), rather than onto [0,Inf)
        self.projection_tol = 1e-6
        #
        # ~~~ Attributes for SSGE and functional training
        self.prior_J   = "please specify"
        self.post_J    = "please specify"
        self.prior_eta = "please specify"
        self.post_eta  = "please specify"
        self.prior_M   = "please specify"
        self.post_M    = "please specify"
        self.measurement_set = None
        self.prior_SSGE      = None
        #
        # ~~~ A hyperparameter needed for the gaussian appxroximation method
        self.post_GP_eta = "please specify"
    #
    # ~~~ Sample according to a "standard normal distribution in the shape of our neural network"
    def sample_from_standard_normal(self):
        for p in self.realized_standard_normal.parameters():
            nn.init.normal_(p)
    #
    # ~~~ Infer device and dtype
    def infer_device_and_dtype(self):
        for layer in self.model_mean:
            if hasattr(layer,"weight"):         # ~~~ the first layer with weights
                device = layer.weight.device
                dtype = layer.weight.dtype
                return device, dtype
    #
    # ~~~ Project the standard deviations to be positive, as in projected gradient descent
    def projection_step( self, soft ):
        with torch.no_grad():
            for p in self.model_std.parameters():
                if not soft:
                    p.data = torch.clamp( p.data, min=self.projection_tol )
                else:
                    try:
                        p.data = self.soft_projection( p.data )
                    except:
                        my_warn("Encountered an error when calling `self.soft_projection(tensor)`... Please ensure that `self.soft_projection`, `self.soft_projection_inv`, and , `self.soft_projection_prime` are all callable on generic tensors.")
                        raise
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
    # ~~~ Initialize the posterior standard deviations to match the standard deviations of a possible prior distribution
    def set_default_uncertainty( self, comparable_to_default_torch_init=False, scale=1.0 ):
        with torch.no_grad():
            if comparable_to_default_torch_init:
                for layer in self.model_std:
                    if isinstance(layer,nn.Linear):
                        std = scale*std_per_layer(layer)
                        layer.weight.data = std * torch.ones_like(layer.weight.data)
                        layer.bias.data = std * torch.ones_like(layer.bias.data)
            else:
                for p in self.model_std.parameters():
                    p.data = scale*std_per_param(p)*torch.ones_like(p.data)
    #
    # ~~~ In Blundell et al. (https://arxiv.org/abs/1505.05424), the chain rule is implemented manually (this is necessary since pytorch doesn't allow in-place operations on the parameters to be included in the graph)
    def apply_chain_rule_for_soft_projection(self):
        with torch.no_grad():
            for p in self.model_std.parameters():
                p.data  = self.soft_projection_inv(p.data)          # ~~~ now, the parameters are \soft_projection = \ln(\exp(\sigma)-1) instead of \sigma
                try:
                    p.grad *= self.soft_projection_prime(p.data)    # ~~~ now, the gradient is \frac{\sigma'}{1+\exp(-\rho)} instead of \sigma'
                except:
                    if p.grad is None:
                        my_warn("`apply_chain_rule_for_soft_projection` operates directly on the `grad` attributes of the parameters. It should be applied *after* `backwards` is called.")
                    raise
    #
    # ~~~ Check that all the posterior standard deviations are positive
    def check_positive(self):
        with torch.no_grad():
            for p in self.model_std.parameters():
                if not p.min() > 0:
                    my_warn("`model_std` contains negative values.")
                    break
    #
    # ~~~ Sample the distribution of Y|X=x,W=w
    def forward(self,x,resample_weights=True):
        #
        # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_normal
        if resample_weights:
            self.sample_from_standard_normal()      # ~~~ this methods re-generates the values of weights and biases in `self.realized_standard_normal` (IID standard normal)
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        self.check_positive()
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
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
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the loss in Bayes by Backprop
    ### ~~~
    #
    # ~~~ Compute ln( f_{Y \mid X,W}(F_\theta(z),x_train,y_train) ) at a point z sampled from the standard MVN distribution ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_likelihood_density( self, X, y ):
        self.last_seen_x = X
        return log_gaussian_pdf( where=y, mu=self(X,resample_weights=False), sigma=self.conditional_std )  # ~~~ Y|X,W is assumed to be normal with mean self(X) and variance self.conditional_std (the latter being a tunable hyper-parameter)
    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_prior_density(self):
        #
        # ~~~ Because the weights and biases are mutually independent, the log prior pdf can be decomposed as a summation \sum_j
        self.check_positive()
        log_prior = 0.
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]        # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            if isinstance( z, nn.modules.linear.Linear ):
                post_mean      =    self.model_mean[j]  # ~~~ the trainable (posterior) means of this layer's parameters
                post_std       =    self.model_std[j]   # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                prior_mean     =    self.prior_mean[j]  # ~~~ the prior means of this layer's parameters
                prior_std      =    self.prior_std[j]   # ~~~ the prior standard deviations of this layer's parameters
                F_theta_of_z   =    post_mean.weight + post_std.weight*z.weight
                log_prior     +=    log_gaussian_pdf( where=F_theta_of_z, mu=prior_mean.weight, sigma=prior_std.weight )
                F_theta_of_z   =    post_mean.bias   +  post_std.bias * z.bias
                log_prior     +=    log_gaussian_pdf( where=F_theta_of_z,  mu=prior_mean.bias,  sigma=prior_std.bias   )
        return log_prior
    #
    # ~~~ Compute \ln( q_\theta(F_\theta(z)) ) at a point z sampled from the standard MVN distribution, where q_\theta is the posterior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_posterior_density(self):
        #
        # ~~~ Because the weights and biases are mutually independent, the log_prior_pdf can be decomposed as a summation \sum_j
        self.check_positive()
        log_posterior = 0.
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]        # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            if isinstance( z, nn.modules.linear.Linear ):
                mean_layer      =    self.model_mean[j] # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer       =    self.model_std[j]  # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                sigma_weight    =    std_layer.weight
                sigma_bias      =    std_layer.bias
                F_theta_of_z    =    mean_layer.weight + sigma_weight*z.weight
                log_posterior  +=    log_gaussian_pdf( where=F_theta_of_z, mu=mean_layer.weight, sigma=sigma_weight )
                F_theta_of_z    =    mean_layer.bias   +  sigma_bias * z.bias
                log_posterior  +=    log_gaussian_pdf( where=F_theta_of_z,  mu=mean_layer.bias,   sigma=sigma_bias  )
        return log_posterior
    #
    # ~~~ Compute the kl divergence between posterior and prior distributions over the network weights
    def weight_kl( self, exact_formula=False ):
        self.check_positive()
        if not exact_formula:
            return self.log_posterior_density() - self.log_prior_density()
        else:
            kl_div = 0.
            #
            # ~~~ Because the weights and biases are mutually independent, the entropy is *additive* like log-density (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties)
            for j in range(self.n_layers):
                post_mean      =    self.model_mean[j]  # ~~~ the trainable (posterior) means of this layer's parameters
                post_std       =    self.model_std[j]   # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                prior_mean     =    self.prior_mean[j]  # ~~~ the prior means of this layer's parameters
                prior_std      =    self.prior_std[j]   # ~~~ the prior standard deviations of this layer's parameters
                if isinstance( post_mean, nn.modules.linear.Linear ):
                    kl_div += diagonal_gaussian_kl( mu_0=post_mean.weight, sigma_0=post_std.weight, mu_1=prior_mean.weight, sigma_1=prior_std.weight )
                    kl_div += diagonal_gaussian_kl(  mu_0=post_mean.bias,   sigma_0=post_std.bias,   mu_1=prior_mean.bias,   sigma_1=prior_std.bias  )
            return kl_div
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the fBNN loss from Sun et al. 2019 (https://arxiv.org/abs/1903.05779)
    ### ~~~
    #
    # ~~~ Sample from the priorly distributed outputs of the network
    def prior_forward(self,x,resample_weights=True):
        #
        # ~~~ If the prior distribution is a Gaussian process, then we just need to sample from the correct Gaussian distribution
        if hasattr(self,"GP"):
            #
            # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_normal
            mu, root_Sigma = self.GP.prior_mu_and_Sigma( x, cholesky=True )  # ~~~ return the cholesky square roots of the covariance matrices
            z = torch.randn( x.shape[0], device=x.device, dtype=x.dtype )
            return mu + (root_Sigma@z).T    # ~~~ mu + Sigma^{-1/2} z \sim N(mu,Sigma); einsum computes this in batches
        #
        # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_normal
        if resample_weights:
            self.sample_from_standard_normal()      # ~~~ this methods re-generates the values of weights and biases in `self.realized_standard_normal` (IID standard normal)
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
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
    # ~~~ Instantiate the SSGE estimator of the prior score, using samples from the prior distribution
    def setup_prior_SSGE(self):
        with torch.no_grad():
            device, dtype = self.infer_device_and_dtype()
            self.measurement_set = self.measurement_set.to( device=device, dtype=dtype )
            #
            # ~~~ Sample from the prior distribution self.prior_M times (and flatten the samples)
            if hasattr(self,"GP"):
                mu, root_Sigma = self.GP.prior_mu_and_Sigma( self.measurement_set, cholesky=True )  # ~~~ return the cholesky square roots of the covariance matrices
                Z = torch.randn( size=( self.prior_M, self.measurement_set.shape[0] ), device=device, dtype=dtype )
                SZ = root_Sigma@Z.T # ~~~ SZ[:,:,j] is the matrix you get from stacking the vectors Sigma_i^{1/2}z_j for i=1,...,self.out_features, where z_j==Z[j] and Sigma_i is the covariance matrix of the model's i-th output
                #
                # ~~~ Sample from the N(mu,Sigma) distribution by taking mu+Sigma^{1/2}z, where z is a sampled from the N(0,I) distribtion
                prior_samples = torch.row_stack([ (mu + SZ[:,:,j].T).flatten() for j in range(self.prior_M) ])
            else:
                prior_samples = torch.row_stack([ self.prior_forward(self.measurement_set).flatten() for _ in range(self.prior_M) ])
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
    # ~~~ Generate a fresh grid of several "points like our model's inputs" from the input domain
    def sample_new_measurement_set(self,n=200):
        #
        # ~~~ In the common case that the inputs are standardized, then standard random normal vectors are "points like our model's inputs"
        device, dtype = self.infer_device_and_dtype()
        self.measurement_set = torch.randn( size=(n,self.in_features), device=device, dtype=dtype )
    #
    # ~~~ Estimate KL_div( posterior output || the prior output ) using the SSGE, assuming we don't have a forula for the density of the outputs
    def functional_kl( self, resample_measurement_set=True ):
        #
        # ~~~ if `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
            device, dtype = self.infer_device_and_dtype()
            self.measurement_set = self.measurement_set.to( device=device, dtype=dtype )
        #
        # ~~~ Prepare for using SSGE to estimate some of the gradient terms
        with torch.no_grad():
            if (self.prior_SSGE is None) or resample_measurement_set:
                self.setup_prior_SSGE()
            posterior_samples = torch.row_stack([ self(self.measurement_set).flatten() for _ in range(self.post_M) ])
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
        # ~~~ Combine all the ingridents as per the chain rule 
        log_posterior_density  =  ( posterior_score_at_yhat @ yhat ).squeeze()  # ~~~ the inner product from the chain rule
        log_prior_density      =  ( prior_score_at_yhat @ yhat ).squeeze()      # ~~~ the inner product from the chain rule            
        return log_posterior_density - log_prior_density
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
            device, dtype = self.infer_device_and_dtype()
            self.measurement_set = self.measurement_set.to( device=device, dtype=dtype )
        n_meas = self.measurement_set.shape[0]
        #
        # ~~~ Assume that the final layer of the architecture is linear, as per the paper's suggestion to take \beta as the parameters of the final layer (very bottom of pg. 4 https://arxiv.org/pdf/2312.17199)
        if not isinstance( self.model_mean[-1] , nn.modules.linear.Linear ):
            raise NotImplementedError('Currently, the only case implemented is the the case from the paper where `\beta` is "the set of parameters in the final neural network layer" (bottom of pg. 4 of the paper).')
        #
        # ~~~ Compute the mean and covariance of a normal distribution approximating that of the (random) output of the network on the measurement set
        out_features = self.model_mean[-1].out_features
        if not approximate_mean:
            #
            # ~~~ Compute the mean and covariance from the paper's equation (14): https://arxiv.org/abs/2312.17199, page 4
            S_sqrt = torch.cat([ p.flatten() for p in self.model_std.parameters() ])  # ~~~ covariance of the joint (diagonal) normal distribution of all network weights is then S_sqrt.diag()**2
            theta_minus_m = S_sqrt*flatten_parameters(self.realized_standard_normal)            # ~~~ theta-m == S_sqrt*z because theta = m+S_sqrt*z
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
            z = torch.cat([             # ~~~ equivalent to `flatten_parameters(self.realized_standard_normal)[ how_many_params_from_not_last_layer: ]`
                    self.realized_standard_normal[-1].weight.flatten(),
                    self.realized_standard_normal[-1].bias.flatten()
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
        # ~~~ Get the mean and covariance of (the Gaussian approximation of) the predicted distribution of yhat
        mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=resample_measurement_set, approximate_mean=approximate_mean )
        if add_stabilizing_noise:
            Sigma_theta += torch.diag( self.post_GP_eta * torch.ones_like(Sigma_theta.diag()) )
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
    # ~~~ A helper function that samples a bunch from the predicted posterior distribution
    def posterior_predicted_mean_and_std( self, x_test, n_samples ):
        with torch.no_grad():
            predictions = torch.column_stack([ self(x_test) for _ in range(n_samples) ])
            std = predictions.std(dim=-1).cpu()             # ~~~ transfer to cpu in order to be able to plot them
            point_estimate = predictions.mean(dim=-1).cpu() # ~~~ transfer to cpu in order to be able to plot them
        return point_estimate, std
