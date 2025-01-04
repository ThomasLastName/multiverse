
"""
Based on code from https://github.com/AntixK/Spectral-Stein-Gradient.

The MIT License (MIT)

Copyright (c) 2020 Anand
Copyright (c) 2024, Triad National Security, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
### ~~~
## ~~~ Reproduce https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/assets/Gaussian.pdf
## ~~~ The returned image should match https://raw.githubusercontent.com/AntixK/Spectral-Stein-Gradient/master/assets/Gaussian.png
### ~~~

import torch
from abc import abstractmethod
from quality_of_life.my_base_utils import my_warn

#
# ~~~ Convert a 1D tensor to a 2D column tensor, but leave every other tensor as is
vertical = lambda x: x.unsqueeze(1) if x.dim()==1 else x


class BaseScoreEstimator:
    #
    # ~~~ Define the RBF kernel
    @staticmethod
    def rbf_kernel( x1, x2, sigma ):
        return torch.exp( -((x1-x2).pow(2).sum(-1)) / (2*sigma**2) ) # / math.sqrt(2*torch.pi) * sigma
    #
    # ~~~ Method that assembles the kernel matrix K
    def gram_matrix( self, x1, x2, sigma ):     # ~~~ e = torch.exp( -((x1.unsqueeze(-2)-x2.unsqueeze(-3)).pow(2).sum(-1)) / (2*sigma**2) ) - torch.exp( -(torch.cdist(x1,x2)/sigma)**2/2 ) \approx 0
        x1, x2 = vertical(x1), vertical(x2)
        return torch.exp( -(torch.cdist(x1,x2)/sigma)**2/2 )
    #
    # ~~~ Method that gram matrix, as well as, the Jacobian matrices which get averaged when computing \beta
    def grad_gram( self, x1, x2, sigma ):
        # """
        # Computes the gradients of the RBF gram matrix with respect
        # to the inputs x1 an x2. It is given by
        # .. math::
        #     \nabla_x1 k(x1, x2) = k(x1, x2) \frac{x1- x2}{\sigma^2}
        #
        #     \nabla_x2 k(x1, x2) = k(x1, x2) -\frac{x1- x2}{\sigma^2}
        #
        # :param x1: (Tensor) [N x D]
        # :param x2: (Tensor) [M x D]
        # :param sigma: (Float) Width of the RBF kernel
        # :return: Gram matrix [N x M],
        #          gradients with respect to x1 [N x M x D],
        #          # gradients with respect to x2 [N x M x D]
        #
        # """
        with torch.no_grad():
            Kxx = self.gram_matrix(x1,x2,sigma)
            x1 = x1.unsqueeze(-2)  # Make it into a column tensor
            x2 = x2.unsqueeze(-3)  # Make it into a row tensor
            diff = (x1 - x2) / (sigma ** 2) # [N x M x D]
            dKxx_dx1 = Kxx.unsqueeze(-1) * (-diff)
            return Kxx, dKxx_dx1
    #
    # ~~~ Method that heuristically chooses the bandwidth sigma for the RBF kernel
    def heuristic_sigma(self,x1,x2):
        # """
        # Uses the median-heuristic for selecting the
        # appropriate sigma for the RBF kernel based
        # on the given samples.
        # The kernel width is set to the median of the
        # pairwise distances between x and xm.
        # :param x: (Tensor) [N x D]
        # :param xm: (Tensor) [M x D]
        # :return:
        # """
        with torch.no_grad():
            x1, x2 = vertical(x1), vertical(x2)
            return torch.cdist(x1,x2).median()
    #
    # ~~~ Placeholder method for the content of __call__(...)
    @abstractmethod
    def compute_score_gradients(self,x):
        raise NotImplementedError
    #
    # ~~~ The `__call__` method just calls `compute_score_gradients`
    def __call__( self, x, *args, **kwargs ):
        return self.compute_score_gradients( x, *args, **kwargs )


class SpectralSteinEstimator(BaseScoreEstimator):
    #
    # ~~~ Allow the user to specify eta for numerical stability as well as J for numerical fidelity
    def __init__( self, samples, eta=None, J=None, sigma=None, h=True, old=False, iterative_avg=False ):
        self.eta = eta
        self.num_eigs = J
        self.samples = vertical(samples)
        self.h = h
        self.M = torch.tensor( samples.size(-2), dtype=samples.dtype, device=samples.device )
        self.sigma = self.heuristic_sigma(self.samples,self.samples) if sigma is None else sigma
        self.eigen_decomposition( old=old, iterative_avg=iterative_avg )
    #
    # ~~~ NEW
    def eigen_decomposition( self, old=False, iterative_avg=False ):
        with torch.no_grad():
            #
            # ~~~ Build the kernel matrix, as well as the associated Jacobians
            xm = self.samples
            #
            # ~~~ Both non-iterative implementations seem to have the exact same memory footprint, but einsum is a lot faster
            if not iterative_avg:
                if old:
                    K, K_Jacobians = self.grad_gram( xm, xm, self.sigma )
                    self.avg_jac = K_Jacobians.mean(dim=0) # [M x D]
                else:
                    K = self.gram_matrix( xm, xm, self.sigma )
                    self.avg_jac = torch.einsum('ij,ijk->ik', K, xm[:, None, :]-xm[None, :, :] ) / (self.sigma**2) / self.M # [M x D]
            #
            # ~~~ Given M,D = samples.shape, the non-iterative implementation (e.g., using einsum) crashes due to memory footprint when D is large. This iterative implentation is more memory effficient
            if iterative_avg:
                K = self.gram_matrix( xm, xm, self.sigma )
                avg_Jac = torch.zeros_like(xm)
                for i in range(xm.shape[1]):
                    diff_i = (xm[:, i].unsqueeze(-1) - xm[:, i].unsqueeze(-2)) / self.sigma**2  # [M x M]
                    K_Jacobian_i = K * (-diff_i)
                    avg_Jac[:, i] = K_Jacobian_i.mean(dim=0)
                self.avg_jac = avg_Jac
            #
            # ~~~ Optionally, K += eta*I for numerical stability
            if self.eta is not None:
                K += self.eta * torch.eye( xm.size(-2), dtype=xm.dtype, device=xm.device )
            #
            # ~~~ Do the actual eigen-decomposition
            if self.num_eigs is None:
                try:
                    eigen_vals, eigen_vecs = torch.linalg.eigh(K) if self.h else torch.linalg.eig(K)
                except RuntimeError:
                    my_warn("There is a bug in the source torch.linalg.eigh code. Specify h=False to use torch.linalg.eig instead")
                    raise
                eigen_vals, eigen_vecs = eigen_vals.flip([0]), eigen_vecs.flip([1])
            else:
                U, s, V  = torch.svd_lowrank( K, q=min(K.shape[0],self.num_eigs) )
                eigen_vals = s
                eigen_vecs = (U+V)/2    # ~~~ by my estimation, because Kxx is symmetric, we actually expect U==V; we are only averaging out the arithmetic errors
            # """
            # instead of:
            # eigen_vals, eigen_vecs = torch.linalg.eig(Kxx)
            # eigen_vals, eigen_vecs = eigen_vals.to(torch.get_default_dtype()), eigen_vecs.to(torch.get_default_dtype())
            # if self.num_eigs is not None:
            #     eigen_vals = eigen_vals[:self.num_eigs]
            #     eigen_vecs = eigen_vecs[:, :self.num_eigs]
            # """
            self.eigen_vals, self.eigen_vecs = eigen_vals, eigen_vecs
    #
    # ~~~ Compute \widehat{Phi}(x)
    def Phi(self,x):
        # """
        # Implements the Nystrom method for approximating the
        # eigenfunction (generalized eigenvectors) for the kernel
        # at x using the M eval_points (x_m). It is given
        # by -
        #
        #  .. math::
        #     phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)
        #
        # :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        # :param eval_points: (Tensor) Sample points from the data of ize M [M x D]
        # :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        # :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        # :param kernel_sigma: (Float) Kernel width
        # :return: Eigenfunction at x [N x M]
        # """
        with torch.no_grad():
            x = vertical(x)
            K_mixed = self.gram_matrix( x, self.samples, self.sigma )
            phi_x =  torch.sqrt(self.M) * K_mixed @ self.eigen_vecs
            phi_x *= 1. / self.eigen_vals
            return phi_x
    #
    # ~~~ Actually estimate \grad \ln(q(x))
    def compute_score_gradients(self,x):
        # """
        # Computes the Spectral Stein Gradient Estimate (SSGE) for the
        # score function. The SSGE is given by
        # 
        # .. math::
        #     \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)
        #
        #     \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)
        #
        #     \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)
        #
        # :param x: (Tensor) Point at which the gradient is evaluated [N x D]
        # :param xm: (Tensor) Samples for the kernel [M x D]
        # :return: gradient estimate [N x D]
        # """
        with torch.no_grad():
            Phi_x = self.Phi(x) # [N x M]
            beta = - torch.sqrt(self.M) * self.eigen_vecs.T @ self.avg_jac
            beta *= (1. / self.eigen_vals.unsqueeze(-1))
            return Phi_x @ beta # [N x D]

