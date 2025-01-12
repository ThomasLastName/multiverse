
import torch
from torch import nn
from abc import abstractmethod
from bnns.SSGE import SpectralSteinEstimator as SSGE
from quality_of_life.my_base_utils import my_warn



### ~~~
## ~~~ Define a BNN with the necessary methods
### ~~~

#
# ~~~ Main class: intended to mimic nn.Sequential
class BayesianModule(nn.Module):
    def __init__(self):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        #
        # ~~~ Attributes for SSGE and functional training
        self.prior_J   = "please specify"
        self.post_J    = "please specify"
        self.prior_eta = "please specify"
        self.post_eta  = "please specify"
        self.prior_M   = "please specify"
        self.post_M    = "please specify"
    #
    # ~~~ Sample from the "variationally distributed" (i.e., learned) outputs of the network
    @abstractmethod
    def forward(self,x):
        raise NotImplementedError("The base class BayesianModule leaves the `forward` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an estimate of `\int ln(f_{Y \mid X,W}(w,X,y)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta` and `f_{Y \mid X,W}(w,X,y)` is the likelihood density
    @abstractmethod
    def log_likelihood_density( self, X, y ):
        # return something like loss_function( f(X;W), y )
        raise NotImplementedError("The base class BayesianModule leaves the `log_likelihood_density` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    # ~~~
    #
    ### ~~~
    ## ~~~ Methods for computing the loss in Bayes by Backprop
    ### ~~~
    #
    # ~~~ Return the exact kl divergence between the variational distribution and the prior distribution
    @abstractmethod
    def exact_weight_kl(self):
        raise NotImplementedError("The base class BayesianModule leaves the `exact_weight_kl` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an esimate of `\int ln(q_\theta(w)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta`
    @abstractmethod
    def log_posterior_density(self):
        raise NotImplementedError("The base class BayesianModule leaves the `log_posterior_density` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Return an esimate of `\int \ln(f_W(w)) q_\theta(w) dw` where `q_\theta(w)` is the variational density with trainable parameters `\theta` and `f_W(w)` is a density function over network weights
    @abstractmethod
    def log_prior_density(self):
        raise NotImplementedError("The base class BayesianModule leaves the `log_prior_density` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
        #
    # ~~~ Compute the kl divergence between posterior and prior distributions
    def weight_kl( self, exact_formula=True ):
        if exact_formula:
            try:
                return self.exact_weight_kl()
            except NotImplementedError:
                if not hasattr(self,"already_warned_that_exact_weight_formula_not_implemented"):
                    my_warn("`exact_weight_kl` method raised a NotImplementedError (specify `weight_kl(exact_formula=False)` instead of `weight_kl(exact_formula=True)` to surpress this warning).")
                    self.already_warned_that_exact_weight_formula_not_implemented = "yup"
        return self.log_posterior_density() - self.log_prior_density()
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
    def sample_new_measurement_set(self,n=200):
        raise NotImplementedError("The base class BayesianModule leaves the `sample_new_measurement_set` method to be implemented in user-defined sub-classes. For a ready-to-use implementation, please see the sub-classes of BayesianModule that are provided with the package.")
    #
    # ~~~ Instantiate the SSGE estimator of the prior score, using samples from the prior distribution
    def setup_prior_SSGE(self):
        with torch.no_grad():
            #
            # ~~~ Sample from the prior distribution self.prior_M times (and flatten the samples)
            prior_samples = self.prior_forward( self.measurement_set, n=self.prior_M ).reshape( self.prior_M, -1 )
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
    # ~~~ Estimate KL_div( variational output || prior output ) using the SSGE, assuming we don't have a forula for the density for the variational distribution of the outputs
    def functional_kl( self, resample_measurement_set=True ):
        #
        # ~~~ if `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
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
