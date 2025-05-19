
import torch
from quality_of_life.my_base_utils import my_warn
from bnns.utils import randmvns
from copy import copy
from tqdm import trange

#
# ~~~ Convert a 1D tensor to a 2D column tensor, but leave every other tensor as is
vertical = lambda x: x.unsqueeze(1) if x.dim()==1 else x

#
# ~~~ Solve Ax=b for x when A is lower-triangular
solve = lambda A, b, upper=False: torch.linalg.solve_triangular( A, b, upper=upper )

#
# ~~~ Main class, with methods for handling all of the linear algebra routines, basically
class RPF_kernel_GP:
    #
    # ~~~
    def __init__(
                self,
                out_features,   # ~~~ the (positive integer) number of output features
                etas,           # ~~~ a list of length out_features; the kernel matrix of j-th output gets a +=etas[j]*I
                bandwidths = None,  # ~~~ a list of length out_features; if None, then it will be inferred heuristically
                scales = None,      # ~~~ a list of length out_features; if None, then it will be taken to all 1's
                means = None    # ~~~ something with a __call__ method satisfying means(x).shape==(x.shape[0],out_features)
            ):
        #
        # ~~~ Features of the problem
        self.out_features = out_features
        self.already_fitted = False
        #
        # ~~~ Kernel hyperparameters
        zero_function = lambda x: torch.zeros( x.shape[0], out_features ).to( device=x.device, dtype=x.dtype )
        self.means = means or zero_function
        self.bandwidths = bandwidths
        self.scales = out_features*[1.] if scales is None else scales
        self.etas = etas
    #
    # ~~~ Build an untrained GPyTorch model
    def build_backend(self,x):
        self.backend = GPYBackend( x, out_features=self.out_features, bandwidths=copy(self.bandwidths), scales=self.scales, etas=self.etas )
    #
    # ~~~ Build a list of covariance matrices (one for each output) K_{i,j} = kernel(x_i,y_j)
    def build_kernel_matrices( self, x, y=None, add_stabilizing_noise=True, check_symmetric=True ):
        #
        # ~~~ Take this as an opportunity to infer a kernel bandwidth, if none was speicied upon initialization
        if y is None: y=x
        x, y = vertical(x), vertical(y)
        dists = torch.cdist(x,y)
        meidan_dist = dists[dists>0].median().item()
        try:    self.bandwidths = [ meidan_dist if bw is None else bw for bw in self.bandwidths ]
        except: self.bandwidths = self.out_features*[torch.cdist(x,y).median().item()]
        #
        # ~~~ Compute those kernel matrices
        un_scaled_kernel_matrices = self.out_features*[ torch.exp(-(dists/self.bandwidths[0])**2/2) ] if len(set(self.bandwidths))==1 else [
                torch.exp( -(dists/self.bandwidths[j])**2/2 )
                for j in range(self.out_features)
            ]
        list_of_kernel_matrices = [ self.scales[j]*un_scaled_kernel_matrices[j] for j in range(self.out_features) ]
        #
        # ~~~ Decide whether or not to add "stabilizing noise"
        if add_stabilizing_noise and check_symmetric:
            m, n = list_of_kernel_matrices[0].shape
            symmetric = torch.allclose( list_of_kernel_matrices[0], list_of_kernel_matrices[0].T ) if m==n else False
            if not symmetric:
                my_warn("Stabilizing noise should not be added to the kernel matrix if the kernel matrix is not symmetric. The supplied argument `add_stabilizing_noise=True` will be ignored.")
                add_stabilizing_noise = False
        if add_stabilizing_noise:
            for j,K in enumerate(list_of_kernel_matrices):
                K += self.etas[j] * torch.eye( x.shape[0], device=x.device, dtype=x.dtype )
        #
        # ~~~ Return whatever we decided
        return torch.stack(list_of_kernel_matrices)
    #
    # ~~~ Compute the means and covariances of the prior GP at x; also, reshape, and apply linear algebra routines, as desired
    def prior_mu_and_Sigma( self, x, add_stabilizing_noise=False, flatten=False, cholesky=False, gpytorch=False ):
        #
        # ~~~ Compute and process the means
        stacked_means = self.means(x)
        μ = stacked_means.flatten() if flatten else stacked_means
        #
        # ~~~ Compute and process the covariance matrices (i.e., kernel matrices)
        if not gpytorch:
            #
            # ~~~ Directly (eagerly) compute the Cholesky factorization of the prior covariance matrix using torch.linalg.cholesky
            list_of_covariance_matrices = self.build_kernel_matrices( x, add_stabilizing_noise=add_stabilizing_noise, check_symmetric=False )
            linalg_routine     = torch.linalg.cholesky if cholesky else lambda K:K
            all_matrices_equal = len(set(self.bandwidths))==len(set(self.etas))==len(set(self.scales))==1  # ~~~ this fails to capture the obvious improvement in the case that only len(set(self.scales))>1
            processed_matrices = self.out_features*[linalg_routine(list_of_covariance_matrices[0])] if all_matrices_equal else linalg_routine(list_of_covariance_matrices)
            Σ = torch.block_diag(*processed_matrices) if flatten else (torch.stack(processed_matrices) if isinstance(processed_matrices,list) else processed_matrices)
        else:
            #
            # ~~~ Employ GPyTorch's implementation using LazyTensors, supposedly more efficient for large datasets
            if not hasattr(self,"backend"): self.build_backend(x)
            _, Σ = self.backend.prior_mu_and_Sigma( x, add_stabilizing_noise=add_stabilizing_noise, flatten=flatten, cholesky=cholesky )
        #
        # ~~~ One final safety check and then return the results
        if flatten: assert μ.ndim==1 and μ.shape==( x.shape[0]*self.out_features, )
        else:       assert μ.ndim==2 and μ.shape==( x.shape[0], self.out_features )
        if flatten: assert Σ.ndim==2 and Σ.shape==( x.shape[0]*self.out_features, x.shape[0]*self.out_features )
        else:       assert Σ.ndim==3 and Σ.shape==( self.out_features, x.shape[0], x.shape[0] )
        return μ, Σ
    #
    # ~~~ Compute K_train^{1/2} and K_train^{-1}@y_train
    def fit( self, x_train, y_train, verbose=True, gpytorch=False ):
        #
        # ~~~ A handful of very basic safety features
        assert y_train.shape[0]==x_train.shape[0], f"The number of training points {x_train.shape[0]} does not match the number of training labels {y_train.shape[0]}."
        assert y_train.ndim==2, f"The training labels must be a 2D tensor. The provided shape{y_train.shape} is not accepted."
        assert y_train.shape[1]==self.out_features, f"The number of columns in the training labels {y_train.shape[1]} does not match the number of output features {self.out_features}."
        if self.already_fitted and verbose: my_warn("This GPR instance has already been fitted. That material will be  overwritten. Use `.fit( x_train, y_train, verbose=False )` to surpress this warning.")
        #
        # ~~~ Employ the Cholesky factorization as in https://gaussianprocess.org/gpml/chapters/RW.pdf
        μ, Σ_sqrt = self.prior_mu_and_Sigma( x=x_train, add_stabilizing_noise=True, flatten=False, cholesky=True, gpytorch=gpytorch )
        y_minus_μ = (y_train-μ).T.unsqueeze(-1)
        alpha = solve( Σ_sqrt.mT, solve(Σ_sqrt,y_minus_μ), upper=True )    # ~~~ L.T \ ( L \ (y-μ) )
        #
        # ~~~ Store the results for later, including the Cholesky factorizations of the prior covariance matrices
        self.x_train = x_train
        self.y_train = y_train
        self.sqrt_Sigma_prior = Σ_sqrt
        self.best_kernel_coefficients = alpha
        self.already_fitted = True
    #
    # ~~~ Get the means and covariance of the posterior distribution of the GP at points x
    def post_mu_and_Sigma( self, x, add_stabilizing_noise=False, flatten=False, cholesky=False, gpytorch=False ):
        #
        # ~~~ Fetch/compute the kernel matrices and means
        μ_PRIOR_test, Σ_PRIOR_test = self.prior_mu_and_Sigma(x)
        if not gpytorch:
            Σ_PRIOR_mixed = self.build_kernel_matrices( self.x_train, x, add_stabilizing_noise=False )
        else:
            #
            # ~~~ Employ GPyTorch's implementation using LazyTensors, supposedly more efficient for large datasets
            if not hasattr(self,"backend"): self.build_backend(x)
            Σ_PRIOR_mixed = self.backend.build_kernel_matrices( self.x_train, x, add_stabilizing_noise=False )
        #
        # ~~~ Compute the posterior means
        μ_POST_test = μ_PRIOR_test + torch.bmm( Σ_PRIOR_mixed.mT, self.best_kernel_coefficients ).squeeze(-1).T   # ~~~ == torch.stack([ μ_PRIOR_test[j] + Σ_PRIOR_mixed[j].T@self.best_kernel_coefficients[j].squeeze() for j in range(self.out_features) ])
        #
        # ~~~ Compute the posterior covariance
        Σ_PRIOR_sqrt = self.sqrt_Sigma_prior
        V = solve( Σ_PRIOR_sqrt, Σ_PRIOR_mixed )
        Σ_POST_test = Σ_PRIOR_test - torch.bmm( V.mT, V )
        #
        # ~~~ Process results if desired before returning them
        if add_stabilizing_noise:
            I = torch.eye( len(x), device=x.device, dtype=x.dtype )
            Σ_POST_test += torch.stack([ eta*I for eta in self.etas ])
        if cholesky: Σ_POST_test = torch.linalg.cholesky(Σ_POST_test)
        if flatten:
            μ_POST_test = μ_POST_test.flatten()
            Σ_POST_test = torch.block_diag(*Σ_POST_test)
        return μ_POST_test, Σ_POST_test
    #
    # ~~~ Return n samples from the posterior distribution at x
    def __call__( self, x, n=1, gpytorch=True ):
        if not self.already_fitted: raise RuntimeError("This GPR instance has not been fitted yet Please call self.fit(x_train,y_train) first.")
        μ, Σ = self.post_mu_and_Sigma( x, add_stabilizing_noise=True, flatten=False, cholesky=True, gpytorch=gpytorch )
        return randmvns( μ, Σ, n=n )
    #
    # ~~~ Return n samples from the prior distribution at x
    def prior_forward( self, x, n=1, gpytorch=True ):
        if not gpytorch:
            #
            # ~~~ Directly ("eagerly") compute the Cholesky factorization of the prior covariance matrix using torch.linalg.cholesky, then return mu + Sigma^{1/2} @ Z_samples
            μ, Σ = self.prior_mu_and_Sigma( x, add_stabilizing_noise=True, flatten=False, cholesky=True )
            return randmvns( μ, Σ, n=n )
        else:
            #
            # ~~~ Employ GPyTorch's implementation, which is considerably more numerically stable
            if not hasattr(self,"backend"): self.build_backend(x)
            return self.backend.prior_forward(x,n)


class simple_mean_zero_RPF_kernel_GP(RPF_kernel_GP):
    def __init__(
                self,
                out_features,
                bw = None,
                scale = 1.,
                eta = 0.001,
            ):
        super().__init__(
                out_features = out_features,
                means = lambda x: torch.zeros( x.shape[0], out_features ).to( device=x.device, dtype=x.dtype ),
                bandwidths = None if (bw is None) else out_features*[bw],
                scales = None if (scale is None) else out_features*[scale],
                etas = out_features*[eta]
            )



### ~~~
## ~~~ Attempt to convert the above to GPyTorch
### ~~~

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


class SingleOutputRBFKernelGP(gpytorch.models.ExactGP):
    def __init__( self, x_train, y_train, likelihood=GaussianLikelihood(), bandwidth=None, scale=1, eta=0.001 ):
        super().__init__( x_train, y_train, likelihood )
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.covar_module.outputscale = scale
        self.covar_module.base_kernel.lengthscale = bandwidth or max( torch.cdist(x_train,x_train).median().item(), 1e-6 )
        self.likelihood.noise = eta
    def forward(self,x):
        return MultivariateNormal( self.mean_module(x), self.covar_module(x) )
    def print_hyperpars(self):
        print("")
        print(f"bandwidth = {self.covar_module.base_kernel.lengthscale.item()}")
        print(f"scale = {self.covar_module.outputscale.item()}")
        print(f"eta = {self.likelihood.noise.item()}")
        print("")

class GPY:
    def __init__( self, out_features, bandwidths=None, scales=None, etas=None ):
        assert out_features>0 and isinstance(out_features,int), f"The number of output features must be  apositive integer, not {out_features}."
        self.out_features = out_features
        self.bandwidths = bandwidths or [None]*out_features
        self.scales = scales or [1.0]*out_features
        self.etas = etas or [0.001]*out_features
        self.models = []
    def fit( self, x_train, y_train, verbose=True ):
        if verbose and len(self.models)>0: my_warn("This GPR instance has already been fitted. That material will be  overwritten. Use `.fit( x_train, y_train, verbose=False )` to surpress this warning.")
        self.models.clear()         # ~~~ self.models = []
        for j in range(self.out_features):
            model = SingleOutputRBFKernelGP(
                x_train = x_train,
                y_train = y_train[:, j],
                bandwidth = self.bandwidths[j],
                scale = self.scales[j],
                eta = self.etas[j]
            )
            model.eval()
            model = model.to( device=x_train.device, dtype=x_train.dtype )
            self.models.append(model)
            self.bandwidths[j] = model.covar_module.base_kernel.lengthscale
    #
    # ~~~ Return n samples from the posterior distribution at x
    def __call__( self, x, n=1 ):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return torch.stack([
                    fitted_gp(x).rsample(torch.Size([n]))
                    for fitted_gp in self.models
                ], dim=-1 ) # ~~~ has shape ( n, len(x), features )
    #
    # ~~~ Return n samples from the posterior distribution at x
    def prior_forward( self, x, n=1 ):
        with torch.no_grad():
            prior_samples = []
            for gp in self.models:
                prior_dist = MultivariateNormal( gp.mean_module(x), gp.covar_module(x) )
                prior_samples.append( prior_dist.rsample(torch.Size([n])) )
            return torch.stack( prior_samples, dim=-1 ) # ~~~ has shape ( n, len(x), features )
    #
    # ~~~ Compute the means and covariances of the prior GP at x; also, reshape, and apply linear algebra routines, as desired
    def prior_mu_and_Sigma( self, x, add_stabilizing_noise=False, flatten=False, cholesky=False ):
        with torch.no_grad():
            μ = torch.zeros( x.shape[0], self.out_features, device=x.device, dtype=x.dtype )
            covariance_matrices = []
            for gp in self.models:
                Σ_lazy = gp.covar_module(x)
                if add_stabilizing_noise: Σ_lazy = gp.likelihood(MultivariateNormal( gp.mean_module(x), Σ_lazy )).lazy_covariance_matrix
                if cholesky: Σ_lazy = Σ_lazy.cholesky()
                covariance_matrices.append(Σ_lazy.to_dense())
            Σ = torch.block_diag(*covariance_matrices) if flatten else (torch.stack(covariance_matrices) if isinstance(covariance_matrices,list) else covariance_matrices)
            return μ, Σ
    #
    # ~~~ Compute the means and covariances of the prior GP at x; also, reshape, and apply linear algebra routines, as desired
    def post_mu_and_Sigma( self, x, add_stabilizing_noise=False, flatten=False, cholesky=False ):
        with torch.no_grad():
            means, covariance_matrices = [], []
            for gp in self.models:
                post_dist = gp(x)
                if add_stabilizing_noise: post_dist = gp.likelihood(post_dist)
                μ_post, Σ_post = post_dist.loc, post_dist.lazy_covariance_matrix
                if cholesky: Σ_lazy = Σ_lazy.cholesky()
                means.append(μ_post)
                covariance_matrices.append(Σ_post.to_dense())
            return torch.stack(means).T, torch.block_diag(*covariance_matrices) if flatten else torch.stack(covariance_matrices)
    #
    # ~~~ Build a list of covariance matrices (one for each output) K_{i,j} = kernel(x_i,y_j)
    def build_kernel_matrices( self, x, y=None, add_stabilizing_noise=True, check_symmetric=True ):
        with torch.no_grad():
            #
            # ~~~ Compute 'em
            if y is None: y=x
            x, y = vertical(x), vertical(y)
            list_of_kernel_matrices = [ gp.covar_module(x,y).evaluate() for gp in self.models ]
            #
            # ~~~ Decide whether or not to add "stabilizing noise"
            if add_stabilizing_noise and check_symmetric:
                m, n = list_of_kernel_matrices[0].shape
                symmetric = torch.allclose( list_of_kernel_matrices[0], list_of_kernel_matrices[0].T ) if m==n else False
                if not symmetric:
                    my_warn("Stabilizing noise should not be added to the kernel matrix if the kernel matrix is not symmetric. The supplied argument `add_stabilizing_noise=True` will be ignored.")
                    add_stabilizing_noise = False
            if add_stabilizing_noise:
                for j,K in enumerate(list_of_kernel_matrices):
                    K += self.etas[j] * torch.eye( x.shape[0], device=x.device, dtype=x.dtype )
            #
            # ~~~ Return whatever we decided
            return torch.stack(list_of_kernel_matrices)

#
# ~~~ Prior GP Only
class GPYBackend(GPY):
    def __init__( self, x, out_features, bandwidths=None, scales=None, etas=None ):
        super().__init__( out_features, bandwidths, scales, etas )
        dummy_x_train = x[0:2,:]
        dummy_y_train = torch.zeros( 2, out_features, device=x.device, dtype=x.dtype )
        self.fit( dummy_x_train, dummy_y_train )    # ~~~ supply dummy training data to get a useless but cheaply fit posterior distibution (we only want the prior)
        for model in self.models: model = model.to( device=x.device, dtype=x.dtype )

#
# ~~~ Another attempt (FOR SOME REASON, SEEMINGLY LESS NUMERICAL STABILITY)
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.mlls import SumMarginalLogLikelihood


def median_pairwise_distance(x):
    """Compute the median of pairwise Euclidean distances for a 2D tensor x."""
    with torch.no_grad():
        x_flat = x.view(x.shape[0], -1)
        dists = torch.cdist(x_flat, x_flat, p=2)
        # Remove diagonal (zeros) and get median
        dists = dists[dists > 0]
        return torch.median(dists)


class SingleOutputGP(gpytorch.models.ExactGP):
    def __init__( self, train_x, train_y, likelihood, bandwidth=None, scale=None ):
        super().__init__( train_x, train_y, likelihood )
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        if bandwidth is not None: self.covar_module.base_kernel.lengthscale = bandwidth
        if scale is not None:     self.covar_module.outputscale = scale
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)


class MultiOutputGP(torch.nn.Module):
    def __init__( self, train_x, train_y, bandwidths=None, scales=None, etas=None ):
        super().__init__()
        self.num_outputs = train_y.shape[1]
        default_bw = median_pairwise_distance(train_x)
        if bandwidths is None: bandwidths = self.num_outputs*[default_bw]
        if scales is None: scales = self.num_outputs*[1]
        if etas is None: etas = self.num_outputs*[0.001]
        self.likelihoods = torch.nn.ModuleList([
            GaussianLikelihood(noise=noise)
            for noise in etas
        ])
        self.models = torch.nn.ModuleList([
            SingleOutputGP( train_x, train_y[:,i], self.likelihoods[i], bandwidth=bandwidths[i], scale=scales[i] )
            for i in range(self.num_outputs)
        ])
    #
    # ~~~ Tune the hyperparameters
    def tune_hyperpars( self, training_iter=50, lr=0.1 ):
        self.models.train()
        self.likelihoods.train()
        optimizer = torch.optim.Adam( self.models.parameters(), lr=lr )
        for i in trange( training_iter, desc="Tuning Hyperparameters" ):
            output = self.models(*[m.train_inputs[0] for m in self.models])
            loss = -self.mll(output, list(m.train_targets for m in self.models))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    #
    # ~~~ 
    def post_mu_and_Sigma( self, x, add_stabilizing_noise=True ):
        self.models.eval()
        self.likelihoods.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvns = [ model(x) for model in self.models ]
            if add_stabilizing_noise: mvns = [ likelihood(mvn) for likelihood, mvn in zip(self.likelihoods,mvns) ]
        means = torch.stack([ mvn.mean for mvn in mvns ])  # ( num_outputs, len(x) )
        covariances = torch.stack([ mvn.covariance_matrix for mvn in mvns ])  # ( num_outputs, len(x), len(x) )
        return means.T, covariances
    #
    # ~~~
    def posterior_samples( self, x, n, add_stabilizing_noise=True ):
        self.models.eval()
        self.likelihoods.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvns = [ model(x) for model in self.models ]
            if add_stabilizing_noise: mvns = [ likelihood(mvn) for likelihood, mvn in zip(self.likelihoods,mvns) ]
            samples = [ mvn.rsample(sample_shape=torch.Size([n])) for mvn in mvns ]
            return torch.stack(samples).permute(1,2,0)  # ( n, len(x), num_outputs )
    #
    # ~~~ 
    def prior_samples(self, x, n, add_stabilizing_noise=False):
        self.eval()
        with torch.no_grad():
            priors = [model.prior_distribution(x) for model in self.models]
            samples = [prior.rsample(torch.Size([n])) for prior in priors]
            if add_stabilizing_noise:
                noise_samples = [
                    torch.randn_like(s) * likelihood.noise.sqrt().item()
                    for s, likelihood in zip(samples, self.likelihoods)
                ]
                samples = [s + eps for s, eps in zip(samples, noise_samples)]
            return torch.stack(samples).permute(1,2,0)  # ( n, len(x), num_outputs )
