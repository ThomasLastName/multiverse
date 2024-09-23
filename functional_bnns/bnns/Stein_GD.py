
import math
import torch
from torch import nn, func, vmap
import copy
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils import log_gaussian_pdf, get_std

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
class SteinEnsemble:
    def __init__( self, list_of_NNs, Optimizer, conditional_std, bw=None ):
        with torch.no_grad():
            #
            # ~~~ Establish basic attributes
            self.models = list_of_NNs    # ~~~ each "particle" is (the parameters of) a neural network
            self.n_models = len(self.models)
            inferred_device = self.models[0][-1].weight.device
            self.conditional_std = conditional_std.to(inferred_device)
            self.bw = bw
            self.optimizers = [ Optimizer(model.parameters()) for model in self.models ]
            #
            # ~~~ Stuff for parallelizing computation of the loss function
            self.all_prior_sigma = torch.tile( torch.cat([ torch.tile(get_std(p),p.shape).flatten() for p in self.models[0].parameters() ]), (self.n_models,1) )
            #
            # ~~~ Weird stuff for parallelizing the forward pass: from https://pytorch.org/tutorials/intermediate/ensembling.html
            base_model = copy.deepcopy(self.models[0])
            base_model = base_model.to("meta")
            def fmodel(params, buffers, x ):
                return func.functional_call( base_model, (params, buffers), (x,) )
            self.fmodel = fmodel
            self.params, self.buffers = func.stack_module_state(self.models)
    #
    # ~~~ Compute A and b for which SVGD prescribes replacing the gradients by `gradients = ( A@gradients + b )/n_particles`
    def compute_affine_transform( self, easy_implementation=False ):
        #
        # ~~~ Flatten each model's parameters into a single vector; row-stack those vectors into a matrix (many more columns than rows)
        all_params = flatten_parameters(self.models) # ~~~ TODO can we just params.flatten(dim=?)?
        #
        # ~~~ In any case, check the kernel bandwidth
        if self.bw is None:
            self.bw = bandwidth_estimator(all_params,all_params)
        #
        # ~~~ Compute the kernel matrix and the "average jacobians"
        if easy_implementation:
            K, grads_of_K = kernel_stuff( all_params, all_params, self.bw ) # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
            sum_grad_info = grads_of_K.sum(axis=0)
        else:
            K = kernel_matrix( all_params, all_params, self.bw )
            sum_grad_info = -torch.einsum('ij,ijk->jk', K, all_params[:, None, :]-all_params[None, :, :] ) / (self.bw**2)
        return K, sum_grad_info
    #
    # ~~~ Zero out all gradients
    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    #
    # ~~~ Compute the loss function, process the gradients (if using SVGD), and update the parameters
    def train_step( self, X, y, stein=True, easy_implementation=False, zero_out_grads=True, vectorized=False ):
        #
        # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
        if easy_implementation:
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
                mu    = self( X, vectorized=vectorized ) # ~~~ TODO this does not work when vectorized=True
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
                K, sum_grad_info = self.compute_affine_transform( easy_implementation=easy_implementation )
                stein_grads = -( K@log_posterior_grads + sum_grad_info ) / len(self.models) # ~~~ take the negative so that pytorch's optimizer's *maximize* the intended objective
                for i, model in enumerate(self.models):
                    set_flat_grads( model, stein_grads[i] )
        #
        # ~~~ Do the update
        for optimizer in self.optimizers:
            #
            # ~~~ TODO is it possible to directoly optimize self.params?
            optimizer.step()
        if zero_out_grads:
            self.zero_grad()
        #
        # ~~~ As far as I can tell, the params used by vmap need to be updated manually like this
        with torch.no_grad():
            self.params, self.buffers = func.stack_module_state(self.models)
    #
    # ~~~ Forward method for the full ensemble
    def __call__( self, X, vectorized=True ):
        if not vectorized:
            return torch.stack([ model(X) for model in self.models ])
        else:
            return vmap( self.fmodel, in_dims=(0,0,None) )( self.params, self.buffers, X )


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