
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


class SingleOutputRBFKernelGP(gpytorch.models.ExactGP):
    def __init__( self, x_train, y_train, bandwidth=None, scale=1, eta=0.001 ):
        likelihood = GaussianLikelihood()
        super().__init__( x_train, y_train, likelihood )
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.covar_module.outputscale = scale
        self.covar_module.base_kernel.lengthscale.item = bandwidth or max( torch.cdist(x_train,x_train).median().item(), 1e-6 )
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
            self.bandwidths[j] = model.covar_module.base_kernel.lengthscale.item()

x_train = torch.randn(40,3)
y_train = torch.randn(40,2)
gpy = GPY( out_features=2, etas=[0.1,0.001] )
gpy.fit(x_train,y_train)
print(gpy.etas) # [0.1, 0.001]
print(gpy.models[0].likelihood.noise)   # tensor([0.0100], grad_fn=<AddBackward0>)
print(gpy.models[1].likelihood.noise)   # tensor([0.0100], grad_fn=<AddBackward0>)

