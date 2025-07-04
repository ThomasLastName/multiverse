import math
import torch
from torch import nn, func, vmap
import copy
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils.math import log_gaussian_pdf, std_per_param
from bnns.utils.handling import (
    my_warn,
    get_flat_grads,
    set_flat_grads,
    nonredundant_copy_of_module_list,
)

kernel_matrix = SSGE_backend().gram_matrix
kernel_stuff = SSGE_backend().grad_gram
bandwidth_estimator = SSGE_backend().heuristic_sigma


#
# ~~~ Compute the mean-zero prior log gaussian density over the model weights
def log_prior_density(model):
    log_prior = 0.0
    for p in model.parameters():
        log_prior += log_gaussian_pdf(
            where=p, mu=torch.zeros_like(p), sigma=std_per_param(p)
        )
    return log_prior


#
# ~~~ Take a list of model; flatten each of their parameters into a single vector; stack those vectors into a matrix
def flatten_parameters(list_of_models):
    return torch.stack(
        [
            torch.cat([p.view(-1) for p in model.parameters()])
            for model in list_of_models
        ]
    )  # ~~~ has shape (len(list_of_models),n_parameters_per_model)


class FModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, params, buffers, x):
        return func.functional_call(self.base_model, (params, buffers), (x,))


loss_fn = nn.MSELoss()


class SteinEnsemble(nn.Module):
    def __init__(self, list_of_NNs, likelihood_std=None, bw=None, Optimizer=None):
        super().__init__()
        with torch.no_grad():
            #
            # ~~~ Establish basic attributes
            self.models = nn.ModuleList(
                list_of_NNs
            )  # ~~~ each "particle" is (the parameters of) a neural network
            self.n_models = len(self.models)
            self.bw = bw
            if Optimizer is not None:
                self.optimizer = Optimizer(
                    self.parameters()
                )  # ~~~ not entirely necessary to have this as an attribute, but such was the case in earlier verions of this code
            if likelihood_std is not None:
                inferred_device = self.models[0][-1].weight.device
                likelihood_std = (
                    likelihood_std
                    if isinstance(likelihood_std, torch.Tensor)
                    else torch.tensor(likelihood_std)
                )
                self.likelihood_std = likelihood_std.to(inferred_device)
            #
            # ~~~ Stuff for parallelizing computation of the loss function
            self.all_prior_sigma = torch.tile(
                torch.cat(
                    [
                        torch.tile(std_per_param(p), p.shape).flatten()
                        for p in self.models[0].parameters()
                    ]
                ),
                (self.n_models, 1),
            )
            self.vmap_failed = False  # ~~~ flag if the generic attempt to vectorize the forward pass has failed
            self.bmm_failed = (
                False  # ~~~ flag if the attempt to use bmm for a sub-routine has failed
            )
            self.iterative_sum = False  # ~~~ if False, then use einsum for a sub-routine, which is faster but more memory intensive then using an iterative sum
            #
            # ~~~ Weird stuff for parallelizing the forward pass: from https://pytorch.org/tutorials/intermediate/ensembling.html
            base_model = copy.deepcopy(self.models[0])
            base_model = base_model.to("meta")
            self.fmodel = FModel(base_model)
            self.params, self.buffers = func.stack_module_state(self.models)
            self.parameters_have_been_updated = True  # ~~~ when true, then it becomes necessary to update self.params and self.buffers

    #
    # ~~~ Compute A and b for which SVGD prescribes replacing the gradients by `gradients = ( A@gradients + b )/n_particles`
    def compute_affine_transform(self, naive_implementation=False, iterative_sum=False):
        #
        # ~~~ Flatten each model's parameters into a single vector; row-stack those vectors into a matrix (many more columns than rows)
        all_params = flatten_parameters(
            self.models
        )  # ~~~ TODO can we just params.flatten(dim=?)?
        #
        # ~~~ In any case, check the kernel bandwidth
        if self.bw is None:
            self.bw = bandwidth_estimator(all_params, all_params)
        #
        # ~~~ Compute the kernel matrix and the "average jacobians"
        if not iterative_sum:
            if naive_implementation:
                K, grads_of_K = kernel_stuff(
                    all_params, all_params, self.bw
                )  # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
                sum_grad_info = grads_of_K.sum(axis=0)
            else:
                K = kernel_matrix(all_params, all_params, self.bw)
                sum_grad_info = -torch.einsum(
                    "ij,ijk->jk", K, all_params[:, None, :] - all_params[None, :, :]
                ) / (self.bw**2)
        #
        # ~~~ Both non-iterative methods (above) seem to have identical memory footprint, though einsum is faster. This method may be have better or worse memory and/or time efficiency. There's no clear preference
        if iterative_sum:
            K = kernel_matrix(all_params, all_params, self.bw)
            sum_grad_info = torch.zeros_like(all_params)
            for i in range(all_params.shape[1]):
                diff_i = (
                    all_params[:, i].unsqueeze(-1) - all_params[:, i].unsqueeze(-2)
                ) / self.bw**2  # [M x M]
                K_Jacobian_i = K * (-diff_i)
                sum_grad_info[:, i] = K_Jacobian_i.sum(dim=0)
        return K, sum_grad_info

    #
    # ~~~ Basic loss
    def mse(self, X, y, vectorized_forward=True):
        predictions = self(X, method=("bmm" if vectorized_forward else "naive"))
        target = torch.tile(y, (self.n_models, 1, 1))
        losses = ((predictions - target) ** 2 / 2).sum(dim=2).mean(dim=1)
        self.parameters_have_been_updated = True  # ~~~ flag that vmap needs to be redefined (redundant but safe if `.backward()` is not called on losses)
        return losses

    #
    # ~~~ Compute ln( f_{Y \mid X,W}(w,x_train,y_train) ) for (the weights w of) every particale, i.e., every net in the ensemble
    def log_likelihood_density(
        self, X, y, naive_implementation=False, forward_method="bmm"
    ):
        #
        # ~~~ Assume that the parameters will be changed (because of a call to `optimizer.step()`) before the next time that the forward method is called
        self.parameters_have_been_updated = True  # ~~~ flag that vmap needs to be redefined (redundant but safe if `.backward()` is not called on losses)
        #
        # ~~~ The most transparent implementation is to just loop over the models
        if naive_implementation:
            #
            # ~~~ torch.stack([ quantity(model) for model in self.models ])
            return torch.stack(
                [
                    log_gaussian_pdf(where=y, mu=model(X), sigma=self.likelihood_std)
                    for model in self.models
                ]
            )
        #
        # ~~~ A more complicated, but faster implementation computes the log likelihood densities for all models "simultaneously," instead of list comprehension
        else:
            #
            # ~~~ Compute the log likelihood densities simultanuously, instead of looping over `model in self.models`
            mu = self(X, method=forward_method)
            n_models, n_data, n_outputs = mu.shape
            where = torch.tile(y, (n_models, 1, 1))
            sigma = torch.tile(self.likelihood_std, (n_models, 1, 1))
            marginal_log_likehoods = -(((where - mu) / sigma) ** 2) / 2 - torch.log(
                math.sqrt(2 * torch.pi) * sigma
            )
            return marginal_log_likehoods.sum(dim=-1).sum(
                dim=-1
            )  # ~~~ a vector of length self.n_models

    #
    # ~~~ Compute \ln( f_W(w) ) for (the weights w of) every particale, i.e., every network in the ensemble
    def log_prior_density(self, naive_implementation=False):
        #
        # ~~~ Assume that the parameters will be changed (because of a call to `optimizer.step()`) before the next time that the forward method is called
        self.parameters_have_been_updated = True  # ~~~ flag that vmap needs to be redefined (redundant but safe if `.backward()` is not called on losses)
        #
        # ~~~ The most transparent implementation is to just loop over the models
        if naive_implementation:
            #
            # ~~~ torch.stack([ quantity(model) for model in self.models ])
            return torch.stack([log_prior_density(model) for model in self.models])
        #
        # ~~~ A more complicated, but faster implementation computes the log likelihood densities for all models "simultaneously," instead of in a loop
        else:
            #
            # ~~~ Compute the log priors densities simultanuously, instead of looping over `model in self.models`
            where = flatten_parameters(self.models)
            sigma = self.all_prior_sigma
            marginal_log_priors = -((where / sigma) ** 2) / 2 - torch.log(
                math.sqrt(2 * torch.pi) * sigma
            )
            return marginal_log_priors.sum(
                dim=-1
            )  # ~~~ a vector of length self.n_models

    #
    # ~~~ Replace the gradients by \widehat{\phi}^*(particle) for each particle (particles are NN's)
    def apply_chain_rule_for_SVGD(self, naive_implementation=False):
        with torch.no_grad():
            #
            # ~~~ TODO use torch.func or, like, vmap or something in place of `get_flat_grads` and `set_flat_grads`
            log_posterior_grads = -torch.stack(
                [get_flat_grads(model) for model in self.models]
            )  # ~~~ has shape (n_models,n_params_in_each_model)
            try:
                K, sum_grad_info = self.compute_affine_transform(
                    naive_implementation=naive_implementation,
                    iterative_sum=self.iterative_sum,
                )
            except:
                #
                # ~~~ In case the the implementation using einsum crashes (presumed due to not enough RAM), then try the more memory-efficient (but slower) impelemntation of the same routine
                my_warn(
                    "Switching to the slower, but more memory-efficient `self.iterative_sum=True`."
                )
                self.iterative_sum = True
                K, sum_grad_info = self.compute_affine_transform(
                    naive_implementation=naive_implementation,
                    iterative_sum=self.iterative_sum,
                )
            #
            # ~~~ Apply the affine transform to the gradients
            stein_grads = -(K @ log_posterior_grads + sum_grad_info) / len(
                self.models
            )  # ~~~ take the negative so that pytorch's optimizer class will *maximize* the intended objective
            for i, model in enumerate(self.models):
                set_flat_grads(model, stein_grads[i])

    #
    # ~~~ Forward method for the full ensemble
    def forward(self, X, method="vmap"):
        #
        # ~~~ Three versions of the forward pass are implemented
        assert method in ["vmap", "bmm", "naive"]
        if (
            (not len(set(len(model) for model in self.models)) == 1)
            and (not self.vmap_failed)
            and (not self.bmm_failed)
        ):
            self.vmap_failed = True
            self.bmm_failed = True
            my_warn(
                "The models in the ensemble do not all have the same number of layers. The vectorized implementation will not work, as a result. Falling back to naive implementation."
            )
        #
        # ~~~ Do the forward pass using `vmap`, which is the fastest method, but not compatible with autograd (basically, you want to use this method for prediction): from https://pytorch.org/tutorials/intermediate/ensembling.html
        if method == "vmap":
            if not self.vmap_failed:
                #
                # ~~~ Try vmap
                try:
                    #
                    # ~~~ As far as I can tell, the params used by vmap need to be updated manually like this
                    if self.parameters_have_been_updated:
                        with torch.no_grad():
                            self.params, self.buffers = func.stack_module_state(
                                self.models
                            )
                        self.parameters_have_been_updated = False
                    #
                    # ~~~ Use vmap with the up-to-date parameters
                    return vmap(self.fmodel, in_dims=(0, 0, None))(
                        self.params, self.buffers, X
                    )
                except:
                    #
                    # ~~~ If the forward pass fails, then try again with `method=bmm`, and don't try `vmap` any more going forward
                    self.vmap_failed = True
                    my_warn("Forward method with `vmap` failed. Falling back to `bmm`.")
            #
            # ~~~ Fall back to bmm
            method = "bmm"
        #
        # ~~~ Do the forward pass using batched matrix multiplication (`torch.bmm`), which is not quite as fast as `vmap`, but is compatible with autograd (basically, you want to use this method for training)
        if method == "bmm":
            if not self.bmm_failed:
                #
                # ~~~ Try bmm
                try:
                    architecture = self.models[0]
                    tiled_yet = False
                    #
                    # ~~~ Loop over the layers
                    for j, layer in enumerate(architecture):
                        #
                        # ~~~ Do `X=layer(X)` in parallel for all the models at once
                        if isinstance(layer, nn.Linear):
                            if not tiled_yet:
                                X = torch.tile(X, (self.n_models, 1, 1))
                                tiled_yet = True  # ~~~ assumes that there are not multiple nn.Unflatten/nn.Flatten layers
                            #
                            # ~~~ Do `X = X@layer.weight.T + layer.bias` in parallel for all the models at once
                            X = torch.bmm(
                                X,
                                torch.stack(
                                    [model[j].weight.T for model in self.models]
                                ),
                            ) + torch.stack(
                                [
                                    torch.tile(model[j].bias, (X.shape[1], 1))
                                    for model in self.models
                                ]
                            )
                        else:
                            #
                            # ~~~ Assumes that every non-linear layer accepts inputs of more or less arbitrary shape (e.g., nn.ReLU)
                            X = torch.stack(
                                [
                                    model[j](X[i] if tiled_yet else X)
                                    for i, model in enumerate(self.models)
                                ]
                            )
                            tiled_yet = True
                    return X
                except:
                    self.bmm_failed = True
                    my_warn(
                        "Failed to vectorize the forward pass. Falling back to the non-vectorized implementation."
                    )
            #
            # ~~~ Fall back to naive
            method = "naive"
        #
        # ~~~ All versions of the forward pass are equivalent to this simple one
        if method == "naive":
            return torch.stack([model(X) for model in self.models])

    #
    # ~~~ https://chatgpt.com/share/6856f65c-c4a0-8001-9e03-aa79878c47f5
    def load_state_dict(self, state_dict, strict=True):
        out = super().load_state_dict(state_dict, strict)
        self.parameters_have_been_updated = True
        return out


#
# ~~~ Given an architecture, initialize n_copies untrained copies with different weights
class SequentialSteinEnsemble(SteinEnsemble):
    def __init__(self, architecture, n_copies, device="cpu", *args, **kwargs):
        self.device = device
        super().__init__(
            list_of_NNs=[
                nonredundant_copy_of_module_list(architecture, sequential=True).to(
                    device
                )
                for _ in range(n_copies)
            ],
            *args,
            **kwargs
        )
