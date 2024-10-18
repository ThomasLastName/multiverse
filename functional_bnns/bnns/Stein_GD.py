
import math
import torch
from torch import nn, func, vmap
import copy
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils import log_gaussian_pdf, get_std

from quality_of_life.my_base_utils import my_warn
from quality_of_life.my_torch_utils import get_flat_grads, set_flat_grads, nonredundant_copy_of_module_list


kernel_matrix = SSGE_backend().gram_matrix
kernel_stuff = SSGE_backend().grad_gram
bandwidth_estimator = SSGE_backend().heuristic_sigma


#
# ~~~ Compute the mean-zero prior log gaussian density over the model weights
def log_prior_density(model):
    log_prior = 0.
    for p in model.parameters():
        log_prior += log_gaussian_pdf( where=p, mu=torch.zeros_like(p), sigma=get_std(p) )
    return log_prior

#
# ~~~ Take a list of model; flatten each of their parameters into a single vector; stack those vectors into a matrix
def flatten_parameters(list_of_models):
    return torch.stack([
            torch.cat([ p.view(-1) for p in model.parameters() ])
            for model in list_of_models
        ])  # ~~~ has shape (len(list_of_models),n_parameters_per_model)

loss_fn = nn.MSELoss()
class SteinEnsemble(nn.Module):
    def __init__( self, list_of_NNs, Optimizer, conditional_std, bw=None ):
        super().__init__()
        with torch.no_grad():
            #
            # ~~~ Establish basic attributes
            self.models = nn.ModuleList(list_of_NNs)        # ~~~ each "particle" is (the parameters of) a neural network
            self.n_models = len(self.models)
            inferred_device = self.models[0][-1].weight.device
            self.conditional_std = conditional_std.to(inferred_device)
            self.bw = bw
            self.optimizer = Optimizer(self.parameters())   # ~~~ not entirely necessary to have this as an attribute, but such was the case in earlier verions of this code
            #
            # ~~~ Stuff for parallelizing computation of the loss function
            self.all_prior_sigma = torch.tile( torch.cat([ torch.tile(get_std(p),p.shape).flatten() for p in self.models[0].parameters() ]), (self.n_models,1) )
            self.failed_to_vectorize = False    # ~~~ flag if the generic attempt to vectorize the forward pass has failed
            #
            # ~~~ Weird stuff for parallelizing the forward pass: from https://pytorch.org/tutorials/intermediate/ensembling.html
            base_model = copy.deepcopy(self.models[0])
            base_model = base_model.to("meta")
            def fmodel(params, buffers, x ):
                return func.functional_call( base_model, (params, buffers), (x,) )
            self.fmodel = fmodel
            self.params, self.buffers = func.stack_module_state(self.models)
            self.parameters_have_been_updated = False   # ~~~ when true, then it becomes necessary to update self.params and self.buffers
    #
    # ~~~ Compute A and b for which SVGD prescribes replacing the gradients by `gradients = ( A@gradients + b )/n_particles`
    def compute_affine_transform( self, naive_implementation=False ):
        #
        # ~~~ Flatten each model's parameters into a single vector; row-stack those vectors into a matrix (many more columns than rows)
        all_params = flatten_parameters(self.models) # ~~~ TODO can we just params.flatten(dim=?)?
        #
        # ~~~ In any case, check the kernel bandwidth
        if self.bw is None:
            self.bw = bandwidth_estimator(all_params,all_params)
        #
        # ~~~ Compute the kernel matrix and the "average jacobians"
        if naive_implementation:
            K, grads_of_K = kernel_stuff( all_params, all_params, self.bw ) # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
            sum_grad_info = grads_of_K.sum(axis=0)
        else:
            K = kernel_matrix( all_params, all_params, self.bw )
            sum_grad_info = -torch.einsum('ij,ijk->jk', K, all_params[:, None, :]-all_params[None, :, :] ) / (self.bw**2)
        return K, sum_grad_info
    #
    # ~~~ Zero out all gradients
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
    #
    # ~~~ Compute the loss function, *and* process the gradients (if using SVGD)
    def compute_loss_and_grads( self, X, y, stein=True, naive_implementation=False, vectorized_forward=True ):
        #
        # ~~~ Record the fact that the parameters have been updated (so that we know to update `vmap` in the __call__ method)
        self.parameters_have_been_updated = True
        #
        # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
        if naive_implementation:
            #
            # ~~~ Straightforward, easy to read implementation
            for model in self.models:
                if not stein:
                    #
                    # ~~~ This is an ordinary neural network ensemble
                    loss = loss_fn(model(X),y) # -log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )loss_fn( model(X), y )
                    loss.backward()
                else:
                    #
                    # ~~~ This is SVGD
                    log_likelihood = log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )
                    log_prior = log_prior_density(model)
                    un_normalized_log_posterior = log_likelihood + log_prior
                    un_normalized_log_posterior.backward()
        else:
            #
            # ~~~ More complicated, but faster implementation
            if stein:
                #
                # ~~~ Compute the log likelihood densities simultanuously, instead of looping over `model in self.models`
                where = torch.tile( y, (self.n_models,1,1) )
                mu    = self( X, method=("bmm" if vectorized_forward else "naive") )
                sigma = torch.tile( self.conditional_std, (self.n_models,1,1) )
                marginal_log_likehoods = -((where-mu)/sigma)**2/2 - torch.log( math.sqrt(2*torch.pi)*sigma )
                joint_log_likelihoods = marginal_log_likehoods.sum(dim=-1).sum(dim=-1)  # ~~~ a vector of length self.n_models
                #
                # ~~~ Compute the log priors densities simultanuously, instead of looping over `model in self.models`
                where = flatten_parameters(self.models)
                mu    = torch.zeros_like(where)
                sigma = self.all_prior_sigma
                marginal_log_priors = -((where-mu)/sigma)**2/2 - torch.log( math.sqrt(2*torch.pi)*sigma )
                joint_log_priors = marginal_log_priors.sum(dim=-1)  # ~~~ a vector of length self.n_models
                un_normalized_log_posteriors = joint_log_likelihoods + joint_log_priors
                # with torch.no_grad():
                #     assert torch.allclose(
                #         un_normalized_log_posteriors,
                #         torch.stack([ log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std ) + log_prior_density(model) for model in self.models ])
                #         )
                un_normalized_log_posteriors.sum().backward()
            else:
                losses = torch.stack([ loss_fn(model(X),y) for model in self.models ])
                losses.sum().backward()
        #
        # ~~~ Replace the gradients by \widehat{\phi}^*(particle) for each particle (particles are NN's)
        if stein:
            with torch.no_grad():
                #
                # ~~~ TODO use torch.func or, like, vmap or something in place of `get_flat_grads` and `set_flat_grads`
                log_posterior_grads = torch.stack([ get_flat_grads(model) for model in self.models ]) # ~~~ has shape (n_models,n_params_in_each_model)
                K, sum_grad_info = self.compute_affine_transform( naive_implementation=naive_implementation )
                stein_grads = -( K@log_posterior_grads + sum_grad_info ) / len(self.models) # ~~~ take the negative so that pytorch's optimizer's *maximize* the intended objective
                for i, model in enumerate(self.models):
                    set_flat_grads( model, stein_grads[i] )
                return un_normalized_log_posterior.detach() if naive_implementation else un_normalized_log_posteriors.detach().mean()
        else:
            return loss.detach() if naive_implementation else losses.detach().mean()
    #
    # ~~~ Compute the loss function, process the gradients (if using SVGD), and update the parameters
    def train_step( self, X, y, stein=True, naive_implementation=False, vectorized_forward=True ):
        loss_without_grad = self.compute_loss_and_grads( X, y, stein=stein, naive_implementation=naive_implementation, vectorized_forward=vectorized_forward )
        #
        # ~~~ Do the update
        self.optimizer.step()
        self.optimizer.zero_grad()
        #
        # ~~~ As far as I can tell, the params used by vmap need to be updated manually like this
        #
        # ~~~ Return the detached loss, for user's refernece
        return loss_without_grad
    #
    # ~~~ Forward method for the full ensemble
    def forward( self, X, method="vmap" ):
        #
        # ~~~ Three versions of the forward pass are implemented
        assert method in ["vmap","bmm","naive"]
        #
        # ~~~ All versions of the forward pass are equivalent to this simple on
        if method=="naive" or self.failed_to_vectorize:
            return torch.stack([ model(X) for model in self.models ])
        #
        # ~~~ Do the forward pass using `vmap`, which is the fastest method, but not compatible with autograd (basically, you want to use this method for prediction): from https://pytorch.org/tutorials/intermediate/ensembling.html
        if method=="vmap":
            if self.parameters_have_been_updated:
                with torch.no_grad():
                    self.params, self.buffers = func.stack_module_state(self.models)
                self.parameters_have_been_updated = False
            return vmap( self.fmodel, in_dims=(0,0,None) )( self.params, self.buffers, X )
        #
        # ~~~ Do the forward pass using batched matrix multiplication (`torch.bmm`), which is not quite as fast as `vmap`, but is compatible with autograd (basically, you want to use this method for training)
        if method=="bmm":
            try:
                architecture = self.models[0]
                tiled_yet = False
                #
                # ~~~ Loop over the layers
                for j,layer in enumerate(architecture):
                    #
                    # ~~~ Don't tile the input until *after* the application of a shaping layer like nn.Unflatten
                    if (not tiled_yet) and (not isinstance(layer,(nn.Unflatten,nn.Flatten))):
                        X = torch.tile( X, (self.n_models,1,1) )
                        tiled_yet = True    # ~~~ assumes that there are not multiple nn.Unflatten/nn.Flatten layers
                    #
                    # ~~~ Do `X=layer(X)` in parallel for all the models at once
                    if isinstance(layer, nn.Linear):
                        #
                        # ~~~ Do `X = X@layer.weight.T + layer.bias` in parallel for all the models at once
                        X = torch.bmm( X, torch.stack([ model[j].weight.T for model in self.models ]) ) + torch.stack([ torch.tile( model[j].bias, (X.shape[1],1) ) for model in self.models ])
                    else:
                        #
                        # ~~~ Assumes that every non-linear layer accepts inputs of more or less arbitrary shape (e.g., nn.ReLU)
                        X = layer(X)
            except:
                self.failed_to_vectorize = True
                self.zero_grad()
                my_warn("Failed to vectorize the forward pass. Falling back to the non-vectorized version.")
                return torch.stack([ model(X) for model in self.models ])
            return X


class SequentialSteinEnsemble(SteinEnsemble):
    def __init__( self, architecture, n_copies, device="cpu", *args, **kwargs ):
        self.device = device
        super().__init__(
                list_of_NNs = [
                    nonredundant_copy_of_module_list( architecture, sequential=True ).to(device)
                    for _ in range(n_copies)
                ],
                *args,
                **kwargs
            )

# class SteinEnsembleDebug:
#     #
#     # ~~~ 
#     def __init__( self, list_of_NNs, Optimizer, conditional_std, bw=None ):
#         self.models = list_of_NNs    # ~~~ each "particle" is (the parameters of) a neural network
#         self.conditional_std = conditional_std
#         self.bw = bw
#         self.optimizers = [ Optimizer(model.parameters()) for model in self.models ]
#     #
#     def kernel_stuff( self, list_of_models, other_list_of_models ):
#         x = flatten_parameters(list_of_models)
#         y = flatten_parameters(other_list_of_models)
#         if self.bw is None:
#             self.bw = bandwidth_estimator(x,y)
#         K, dK = kernel_stuff(x,y,self.bw)
#         return K, dK    # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
#     #
#     def train_step(self,X,y):
#         #
#         # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
#         for model in self.models:
#             log_likelihood = log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )
#             log_prior = 0. #log_prior_density(model)
#             negative_un_normalized_log_posterior = -(log_likelihood + log_prior)
#             negative_un_normalized_log_posterior.backward()
#         #
#         # ~~~ Apply the affine transformation
#         with torch.no_grad():
#             log_posterior_grads = torch.stack([ get_flat_grads(model) for model in self.models ]) # ~~~ has shape (n_models,n_params_in_each_model)
#             K, grads_of_K = self.kernel_stuff( self.models, self.models )
#             # K, grads_of_K = torch.eye( len(self.models), device=K.device, dtype=K.dtype ), torch.zeros_like(grads_of_K)
#             stein_grads = ( K@log_posterior_grads + grads_of_K.sum(axis=0) ) / len(self.models)
#             for i, model in enumerate(self.models):
#                 set_flat_grads( model, stein_grads[i] )
#         #
#         # ~~~
#         for optimizer in self.optimizers:
#             optimizer.step()
#             optimizer.zero_grad()
#         # return K, grads_of_K
#     #
#     # ~~~ View the full ensemble
#     def __call__(self,x):
#         return torch.column_stack([ model(x) for model in self.models ])




# class SequentialSteinEnsembleDebug(SteinEnsembleDebug):
#     def __init__( self, architecture, n_copies, *args, **kwargs ):
#         some_device = "cuda" if torch.cuda.is_available() else "cpu"
#         super().__init__(
#             list_of_NNs = [
#                 nonredundant_copy_of_module_list( architecture, sequential=True ).to(some_device)
#                 for _ in range(n_copies)
#             ],
#             *args,
#             **kwargs
#         )