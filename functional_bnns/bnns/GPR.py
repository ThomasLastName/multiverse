
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
# ~~~ Check whether all of the matrices in a list are equal
def all_matrices_equal(matrix_list):
    first_matrix = matrix_list[0]
    return all(torch.equal(first_matrix, matrix) for matrix in matrix_list)

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
    def build_kernel_matrices( self, x, y=None, add_stabilizing_noise=True ):
        #
        # ~~~ Take this as an opportunity to infer a kernel bandwidth, if none was speicied upon initialization
        if self.bandwidths is None: self.set_bandwidth_based_on_data(x,y)
        #
        # ~~~ Compute 'em
        if y is None: y=x
        x, y = vertical(x), vertical(y)
        dists = torch.cdist(x,y)
        list_of_kernel_matrices = [
                    self.scales[j] * torch.exp( -(dists/self.bandwidths[j])**2/2 )
                    for j in range(self.out_features)
                ]
        #
        # ~~~ Decide whether or not to add "stabilizing noise"
        kernel_matrix_is_symmetric = (y is None) or torch.equal(x,y)
        add_stabilizing_noise = (kernel_matrix_is_symmetric and add_stabilizing_noise)
        if add_stabilizing_noise:
            for j,K in enumerate(list_of_kernel_matrices):
                K += self.etas[j] * torch.eye( x.shape[0], device=x.device, dtype=x.dtype )
        #
        # ~~~ Return whatevver we decided
        return torch.stack(list_of_kernel_matrices)
    #
    # ~~~ Shape shit and apply linear algebra routines, as desired
    def process_means_and_covariances_at_x( self, x, stacked_means, list_of_covariance_matrices, flatten=False, inv=False, cholesky=True ):
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
            for K in list_of_covariance_matrices:
                if not K.shape==K.T.shape:
                    raise ValueError("`cholesky=True` (i.e., taking a cholesky decomposition) requires that all covariance matrices are symmetric. The ones supplied weren't even square!")
            linalg_routine = square_root_of_inverse if inv else torch.linalg.cholesky
        else:
            linalg_routine = torch.linalg.inv if inv else lambda K:K
        #
        # ~~~ Either stack the covariance matrices of each output feature into a "third order tensor", or form a block diagonal matrix out of them
        stack_or_block = lambda list_of_matrices: torch.block_diag(*list_of_matrices) if flatten else torch.stack(list_of_matrices)
        SIGMA = stack_or_block(
                self.out_features*[linalg_routine(list_of_covariance_matrices[0])]
                if all_matrices_equal(list_of_covariance_matrices) else # ~~~ if they're all the same, don't bother running `linalg_routine` on all of them
                [ linalg_routine(K) for K in list_of_covariance_matrices ]
            )
        if flatten:
            assert SIGMA.ndim  == 2
            assert SIGMA.shape == ( x.shape[0]*self.out_features, x.shape[0]*self.out_features )
        else:
            assert SIGMA.ndim  == 3
            assert SIGMA.shape == ( self.out_features, x.shape[0], x.shape[0] )
        return MU, SIGMA
    #
    # ~~~ Get the means and covariance of the prior distribution of the GP at points x
    def prior_mu_and_Sigma( self, x, inv=False, flatten=False, cholesky=False ):
        MU = self.means(x)
        SIGMA = self.build_kernel_matrices(x)
        return self.process_means_and_covariances_at_x(
                x = x,
                stacked_means = MU,
                list_of_covariance_matrices = SIGMA,
                flatten = flatten,
                inv = inv,
                cholesky = cholesky
            )
    #
    # ~~~
    def fit( self, x_train, y_train ):
        assert y_train.shape[0]==x_train.shape[0], f"The number of training points {x_train.shape[0]} does not match the number of training labels {y_train.shape[0]}."
        assert y_train.ndim==2, f"The training labels must be a 2D tensor. The provided shape{y_train.shape} is not accepted."
        assert y_train.shape[1]==self.out_features, f"The number of columns in the training labels {y_train.shape[1]} does not match the number of output features {self.out_features}."
        if self.already_fitted: raise RuntimeError("This GPR instance has already been fitted.")
        MU = self.means(x_train)
        SIGMA = self.build_kernel_matrices(x_train)
        MU, SIGMA_SQRT = self.process_means_and_covariances_at_x(
                x = x_train,
                stacked_means = MU,
                list_of_covariance_matrices = SIGMA,
                flatten = False,
                inv = False,
                cholesky = True
            )
        with support_for_progress_bars():
            if self.out_features>1: pbar = tqdm( total=self.out_features, ascii=" >=", desc="Computing Posteriors of Each Output Feature" )
            alpha = []
            for j in range(self.out_features):
                #
                # ~~~ Based on the pseudo-code for algorithm 2.1 in https://gaussianprocess.org/gpml/chapters/RW.pdf
                L = SIGMA_SQRT[j]
                y_minus_mu = (y_train[:,j] - MU[:,j]).reshape(-1,1)
                alpha.append(solve( L.T, solve(L,y_minus_mu), upper=True ))
                if self.out_features>1: pbar.update()
            if self.out_features>1: pbar.close()
        #
        # ~~~ Store the results for later
        self.x_train = x_train
        self.y_train = y_train
        self.SIGMA_PRIOR_SQRT = SIGMA_SQRT
        self.best_kernel_coefficients = torch.stack(alpha)
        self.already_fitted = True
    #
    # ~~~ Get the means and covariance of the prior distribution of the GP at points x
    def post_mu_and_Sigma(self,x):
        MU_PRIOR_test = self.means(x)
        SIGMA_PRIOR_test = self.build_kernel_matrices( x, add_stabilizing_noise=False )
        SIGMA_PRIOR_mixed = self.build_kernel_matrices( self.x_train, x, add_stabilizing_noise=False )
        MU_POST_test = MU_PRIOR_test + torch.bmm( SIGMA_PRIOR_mixed.permute(0,2,1) , self.best_kernel_coefficients ).squeeze(dim=-1).T  # ~~~ == torch.stack([ MU_PRIOR_test[j] + SIGMA_PRIOR_mixed[j].T@self.best_kernel_coefficients[j].squeeze() for j in range(self.out_features) ])
        V = torch.stack([ solve( self.SIGMA_PRIOR_SQRT[j], SIGMA_PRIOR_mixed[j]) for j in range(self.out_features) ])
        SIGMA_POST_test = SIGMA_PRIOR_test - torch.stack([ v.T@v for v in V ])
        return  MU_POST_test, SIGMA_POST_test
    #
    # ~~~
    def __call__(self,x,n=1):
        return randmvns( *self.post_mu_and_Sigma(x), n=n )


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
