import torch
from abc import abstractmethod


class BaseScoreEstimator:
    #
    # ~~~ Define the RBF kernel
    @staticmethod
    def rbf_kernel(x1, x2, sigma):
        return torch.exp(
            -((x1 - x2).pow(2).sum(-1)) / (2 * sigma**2)
        )  # / math.sqrt(2*torch.pi) * sigma

    #
    # ~~~ Method that assembles the kernel matrix K
    def gram_matrix(self, x1, x2, sigma):
        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor
        return self.rbf_kernel(x1, x2, sigma)

    #
    # ~~~ Method that gram matrix, as well as, the Jacobian matrices which get averaged when computing \beta
    def grad_gram(self, x1, x2, sigma):
        with torch.no_grad():
            Kxx = self.gram_matrix(x1, x2, sigma)
            x1 = x1.unsqueeze(-2)  # Make it into a column tensor
            x2 = x2.unsqueeze(-3)  # Make it into a row tensor
            diff = (x1 - x2) / (sigma**2)  # [N x M x D]
            dKxx_dx1 = Kxx.unsqueeze(-1) * (-diff)
            return Kxx, dKxx_dx1

    #
    # ~~~ Method that heuristically chooses the bandwidth sigma for the RBF kernel
    def heuristic_sigma(self, x1, x2):
        with torch.no_grad():
            x1 = x1.unsqueeze(-2)  # Make it into a column tensor
            x2 = x2.unsqueeze(-3)  # Make it into a row tensor
            pdist_mat = ((x1 - x2) ** 2).sum(dim=-1).sqrt()  # [N x M]
            kernel_width = torch.median(torch.flatten(pdist_mat))
            return kernel_width

    #
    # ~~~ Placeholder method for the content of __call__(...)
    @abstractmethod
    def compute_score_gradients(self, x):
        raise NotImplementedError

    #
    # ~~~ The `__call__` method just calls `compute_score_gradients`
    def __call__(self, x):
        return self.compute_score_gradients(x)


class SpectralSteinEstimator(BaseScoreEstimator):
    #
    # ~~~ Allow the user to specify eta for numerical stability as well as J for numerical fidelity
    def __init__(self, samples, eta=None, J=None, sigma=None):
        self.eta = eta
        self.num_eigs = J
        self.samples = samples
        self.M = torch.tensor(
            samples.size(-2), dtype=samples.dtype, device=samples.device
        )
        self.sigma = (
            self.heuristic_sigma(self.samples, self.samples) if sigma is None else sigma
        )
        self.eigen_decomposition()

    #
    # ~~~ NEW
    def eigen_decomposition(self):
        with torch.no_grad():
            #
            # ~~~ Build the kernel matrix, as well as the associated Jacobians
            xm = self.samples
            self.K, self.K_Jacobians = self.grad_gram(xm, xm, self.sigma)
            self.avg_jac = self.K_Jacobians.mean(dim=-3)  # [M x D]
            #
            # ~~~ Optionally, K += eta*I for numerical stability
            if self.eta is not None:
                self.K += self.eta * torch.eye(
                    xm.size(-2), dtype=xm.dtype, device=xm.device
                )
            #
            # ~~~ Do the actual eigen-decomposition
            if self.num_eigs is None:
                eigen_vals, eigen_vecs = torch.linalg.eigh(self.K)
                eigen_vals, eigen_vecs = eigen_vals.flip([0]), eigen_vecs.flip([1])
            else:
                U, s, V = torch.svd_lowrank(
                    self.K, q=min(self.K.shape[0], self.num_eigs)
                )
                eigen_vals = s
                eigen_vecs = (
                    U + V
                ) / 2  # ~~~ by my estimation, because Kxx is symmetric, we actually expect U==V; we are only averaging out the arithmetic errors
            self.eigen_vals, self.eigen_vecs = eigen_vals, eigen_vecs


torch.manual_seed(1234)
M = 3000  # ~~~ will be implicity rounded *down* the the nearest square number: int(sqrt(M))**2
eta = 0.0095
make_gif = False
D = 10
n_test = 500
device = "cuda" if torch.cuda.is_available() else "cpu"


test_points = torch.randn(n_test, D, device=device)
samples = torch.randn(M, D, device=device)
xm = samples
x = test_points
self = SpectralSteinEstimator(eta=eta, samples=xm)
