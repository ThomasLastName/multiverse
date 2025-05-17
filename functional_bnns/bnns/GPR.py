
import torch
from tqdm import tqdm
from quality_of_life.my_base_utils import my_warn, support_for_progress_bars
from bnns.utils import randmvns

#
# ~~~ Compute the Cholesky decopmosition of A^{-1}... with a bunch of `except` clauses to handle numerical instability
def square_root_of_inverse(K):
    #
    # ~~~ Try the thing that we would normally want to do
    try:
        root_K_inv = torch.linalg.cholesky(torch.linalg.inv(K))
    except:
        #
        # ~~~ In case of numerical instabiltiy, try this instead... just in case maybe it's more robust?
        try:
            root_K_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(K)))
            print("Having to use `cholesky_inverse`")
        #
        # ~~~ If that, still, doesn't work, conclude that we either gotta add noise or use a higher degree of numerical precision
        except torch._C._LinAlgError:
            message = 'pythorch is having trouble taking the cholesky inverse of the covariance matrix of the prior GP. Perhaps, consider adding more "stabilizing noise" (i.e., K + eta*I) by increasing the value of the parameter for this class'
            if not K.dtype==torch.double:
                message += ' and/or increasing the numerical precision by using torch.double if using torch.float currently'
            my_warn(message)
        #
        # ~~~ Raise the original exception
        raise
    #
    # ~~~ Return the result, if it worked
    return root_K_inv

#
# ~~~ Compute torch.stack([ C[j] - B[j]@A[j]@B[j].T for j in range(d) ]) for lists of matrices; see the script `test_bmm.py`
def compute_C_minus_BABt(A,B,C):
    return C - torch.bmm( torch.bmm(B, A), B.transpose(1, 2) )

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
    def prior_mu_and_Sigma( self, x, flatten=False, inv=False, cholesky=False ):
        #
        # ~~~ Compute 'em
        stacked_means = self.means(x)
        list_of_covariance_matrices = self.build_kernel_matrices( x, add_stabilizing_noise=True, check_symmetric=False )
        #
        # ~~~ We don't do much to the means; either flatten them, or don't
        MU = stacked_means.flatten() if flatten else stacked_means
        if flatten:
            assert MU.ndim  == 1
            assert MU.shape == ( x.shape[0]*self.out_features, )
        else:
            assert MU.ndim  == 2
            assert MU.shape == ( x.shape[0], self.out_features )
        #
        # ~~~ Specify whether to take the inverse and/or (afterwards) to take the cholesky square root of the covariance matrices
        if cholesky:
            linalg_routine = square_root_of_inverse if inv else torch.linalg.cholesky
        else:
            linalg_routine = torch.linalg.inv if inv else lambda K:K
        #
        # ~~~ Either stack the covariance matrices of each output feature into a "third order tensor", or form a block diagonal matrix out of them
        all_matrices_equal = len(set(self.bandwidths))==len(set(self.etas))==len(set(self.scales))==1  # ~~~ this fails to capture the obvious improvement in the case that only len(set(self.scales))>1
        processed_matrices = self.out_features*[linalg_routine(list_of_covariance_matrices[0])] if all_matrices_equal else [ linalg_routine(K) for K in list_of_covariance_matrices ]
        SIGMA = torch.block_diag(*processed_matrices) if flatten else torch.stack(processed_matrices)
        if flatten:
            assert SIGMA.ndim  == 2
            assert SIGMA.shape == ( x.shape[0]*self.out_features, x.shape[0]*self.out_features )
        else:
            assert SIGMA.ndim  == 3
            assert SIGMA.shape == ( self.out_features, x.shape[0], x.shape[0] )
        return MU, SIGMA
    #
    # ~~~ Compute K_train^{1/2} and K_train^{-1}@y_train
    def fit( self, x_train, y_train ):
        #
        # ~~~ A handful of very basic safety features
        assert y_train.shape[0]==x_train.shape[0], f"The number of training points {x_train.shape[0]} does not match the number of training labels {y_train.shape[0]}."
        assert y_train.ndim==2, f"The training labels must be a 2D tensor. The provided shape{y_train.shape} is not accepted."
        assert y_train.shape[1]==self.out_features, f"The number of columns in the training labels {y_train.shape[1]} does not match the number of output features {self.out_features}."
        if self.already_fitted: raise RuntimeError("This GPR instance has already been fitted.")
        #
        # ~~~ Employ the Cholesky factorization as in https://gaussianprocess.org/gpml/chapters/RW.pdf
        MU, SIGMA_SQRT = self.prior_mu_and_Sigma( x=x_train, flatten=False, inv=False, cholesky=True )
        y_minus_mu = (y_train-MU).T.unsqueeze(-1)
        alpha = solve( SIGMA_SQRT.mT, solve(SIGMA_SQRT,y_minus_mu), upper=True )    # ~~~ L.T \ ( L \ (y-mu) )
        #
        # ~~~ Store the results for later, including the Cholesky factorizations of the prior covariance matrices
        self.x_train = x_train
        self.y_train = y_train
        self.SIGMA_PRIOR_SQRT = SIGMA_SQRT
        self.best_kernel_coefficients = alpha
        self.already_fitted = True
    #
    # ~~~ Get the means and covariance of the prior distribution of the GP at points x
    def post_mu_and_Sigma(self,x):
        MU_PRIOR_test, SIGMA_PRIOR_test = self.prior_mu_and_Sigma(x)
        SIGMA_PRIOR_mixed = self.build_kernel_matrices( self.x_train, x, add_stabilizing_noise=False )
        MU_POST_test = MU_PRIOR_test + torch.bmm( SIGMA_PRIOR_mixed.mT, self.best_kernel_coefficients ).squeeze(-1).T   # ~~~ == torch.stack([ MU_PRIOR_test[j] + SIGMA_PRIOR_mixed[j].T@self.best_kernel_coefficients[j].squeeze() for j in range(self.out_features) ])
        V = solve( self.SIGMA_PRIOR_SQRT, SIGMA_PRIOR_mixed )
        SIGMA_POST_test = SIGMA_PRIOR_test - torch.bmm( V.mT, V )
        return MU_POST_test, SIGMA_POST_test
    #
    # ~~~ Return n samples from the posterior distribution at x
    def __call__(self,x,n=1):
        if not self.already_fitted: raise RuntimeError("This GPR instance has not been fitted yet Please call self.fit(x_train,y_train) first.")
        return randmvns( *self.post_mu_and_Sigma(x), n=n )
    #
    # ~~~ Return n samples from the prior distribution at x
    def prior_forward(self,x,n=1):
        return randmvns( *self.prior_mu_and_Sigma(x), n=n )


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
