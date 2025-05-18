
import torch
from quality_of_life.my_base_utils import my_warn
from bnns.utils import randmvns

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
                means,  # ~~~ something with a __call__ method satisfying means(x).shape==(x.shape[0],out_features)
                etas,   # ~~~ a list of length out_features; the kernel matrix of j-th output gets a +=etas[j]*I
                bandwidths = None,  # ~~~ a list of length out_features; if None, then it will be inferred heuristically
                scales = None       # ~~~ a list of length out_features; if None, then it will be taken to all 1's
            ):
        #
        # ~~~ Features of the problem
        self.out_features = out_features
        self.already_fitted = False
        #
        # ~~~ Kernel hyperparameters
        self.means = means
        self.bandwidths = bandwidths
        self.scales = out_features*[1.] if scales is None else scales
        self.etas = etas
    #
    # ~~~ Infer a kernel bandwidth from the data heuristically
    def set_bandwidth_based_on_data(self,x,y):
        if y is None: y=x
        x, y = vertical(x), vertical(y)
        self.bandwidths = self.out_features*[torch.cdist(x,y).median().item()]
    #
    # ~~~ Build a list of covariance matrices (one for each output) K_{i,j} = kernel(x_i,y_j)
    def build_kernel_matrices( self, x, y=None, add_stabilizing_noise=True, check_symmetric=True ):
        #
        # ~~~ Take this as an opportunity to infer a kernel bandwidth, if none was speicied upon initialization
        if self.bandwidths is None: self.set_bandwidth_based_on_data(x,y)
        #
        # ~~~ Compute 'em
        if y is None: y=x
        x, y = vertical(x), vertical(y)
        dists = torch.cdist(x,y)
        un_scaled_kernel_matrices = [
                    torch.exp( -(dists/self.bandwidths[j])**2/2 )
                    for j in range(self.out_features)
                ] if len(set(self.bandwidths))==1 else self.out_features*[ torch.exp(-(dists/self.bandwidths[0])**2/2)]
        list_of_kernel_matrices = [ self.scales[j]*un_scaled_kernel_matrices[j] for j in range(self.out_features) ]
        #
        # ~~~ Decide whether or not to add "stabilizing noise"
        if add_stabilizing_noise and check_symmetric:
            if not torch.equal( list_of_kernel_matrices[0], list_of_kernel_matrices[0].T ):
                my_warn("Stabilizing noise should not be added to the kernel matrix if the kernel matrix is not symmetric. The supplied argument `add_stabilizing_noise=True` will be ignored.")
                add_stabilizing_noise = False
        if add_stabilizing_noise:
            for j,K in enumerate(list_of_kernel_matrices):
                K += self.etas[j] * torch.eye( x.shape[0], device=x.device, dtype=x.dtype )
        #
        # ~~~ Return whatevver we decided
        return torch.stack(list_of_kernel_matrices)
    #
    # ~~~ Compute the means and covariances, reshape, and apply linear algebra routines, as desired
    def prior_mu_and_Sigma( self, x, flatten=False, cholesky=False ):
        #
        # ~~~ Compute 'em
        stacked_means = self.means(x)
        list_of_covariance_matrices = self.build_kernel_matrices( x, add_stabilizing_noise=True, check_symmetric=False )
        #
        # ~~~ We don't do much to the means; either flatten them, or don't
        μ = stacked_means.flatten() if flatten else stacked_means
        if flatten: assert μ.ndim==1 and μ.shape==( x.shape[0]*self.out_features, )
        else:       assert μ.ndim==2 and μ.shape==( x.shape[0], self.out_features )
        #
        # ~~~ Either stack the covariance matrices of each output feature into a "third order tensor", or form a block diagonal matrix out of them
        linalg_routine     = torch.linalg.cholesky if cholesky else lambda K:K
        all_matrices_equal = len(set(self.bandwidths))==len(set(self.etas))==len(set(self.scales))==1  # ~~~ this fails to capture the obvious improvement in the case that only len(set(self.scales))>1
        processed_matrices = self.out_features*[linalg_routine(list_of_covariance_matrices[0])] if all_matrices_equal else linalg_routine(list_of_covariance_matrices)
        Σ = torch.block_diag(*processed_matrices) if flatten else torch.stack(processed_matrices)
        if flatten: assert Σ.ndim==2 and Σ.shape==( x.shape[0]*self.out_features, x.shape[0]*self.out_features )
        else:       assert Σ.ndim==3 and Σ.shape==( self.out_features, x.shape[0], x.shape[0] )
        return μ, Σ
    #
    # ~~~ Compute K_train^{1/2} and K_train^{-1}@y_train
    def fit( self, x_train, y_train, verbose=True ):
        #
        # ~~~ A handful of very basic safety features
        assert y_train.shape[0]==x_train.shape[0], f"The number of training points {x_train.shape[0]} does not match the number of training labels {y_train.shape[0]}."
        assert y_train.ndim==2, f"The training labels must be a 2D tensor. The provided shape{y_train.shape} is not accepted."
        assert y_train.shape[1]==self.out_features, f"The number of columns in the training labels {y_train.shape[1]} does not match the number of output features {self.out_features}."
        if self.already_fitted and verbose: my_warn("This GPR instance has already been fitted. That material will be  overwritten. Use `.fit( x_train, y_train, verbose=False )` to surpress this warning.")
        #
        # ~~~ Employ the Cholesky factorization as in https://gaussianprocess.org/gpml/chapters/RW.pdf
        μ, Σ_sqrt = self.prior_mu_and_Sigma( x=x_train, flatten=False, cholesky=True )
        y_minus_mu = (y_train-μ).T.unsqueeze(-1)
        alpha = solve( Σ_sqrt.mT, solve(Σ_sqrt,y_minus_mu), upper=True )    # ~~~ L.T \ ( L \ (y-mu) )
        #
        # ~~~ Store the results for later, including the Cholesky factorizations of the prior covariance matrices
        self.x_train = x_train
        self.y_train = y_train
        self.sqrt_Sigma_prior = Σ_sqrt
        self.best_kernel_coefficients = alpha
        self.already_fitted = True
    #
    # ~~~ Get the means and covariance of the prior distribution of the GP at points x
    def post_mu_and_Sigma( self, x, flatten=False, cholesky=False ):
        μ_PRIOR_test, Σ_PRIOR_test = self.prior_mu_and_Sigma(x)
        Σ_PRIOR_mixed = self.build_kernel_matrices( self.x_train, x, add_stabilizing_noise=False )
        μ_POST_test = μ_PRIOR_test + torch.bmm( Σ_PRIOR_mixed.mT, self.best_kernel_coefficients ).squeeze(-1).T   # ~~~ == torch.stack([ μ_PRIOR_test[j] + Σ_PRIOR_mixed[j].T@self.best_kernel_coefficients[j].squeeze() for j in range(self.out_features) ])
        Σ_PRIOR_sqrt = self.sqrt_Sigma_prior
        V = solve( Σ_PRIOR_sqrt, Σ_PRIOR_mixed )
        Σ_POST_test = Σ_PRIOR_test - torch.bmm( V.mT, V )
        if cholesky: Σ_POST_test = torch.linalg.cholesky(Σ_POST_test)
        if flatten:
            μ_POST_test = μ_POST_test.flatten()
            Σ_POST_test = torch.block_diag(*Σ_POST_test)
        return μ_POST_test, Σ_POST_test
    #
    # ~~~ Return n samples from the posterior distribution at x
    def __call__(self,x,n=1):
        if not self.already_fitted: raise RuntimeError("This GPR instance has not been fitted yet Please call self.fit(x_train,y_train) first.")
        return randmvns( *self.post_mu_and_Sigma(x,cholesky=True), n=n )
    #
    # ~~~ Return n samples from the prior distribution at x
    def prior_forward(self,x,n=1):
        return randmvns( *self.prior_mu_and_Sigma(x,cholesky=True), n=n )


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
