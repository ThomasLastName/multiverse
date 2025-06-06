
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from tqdm import trange
from bnns.SequentialGaussianBNN import SequentialGaussianBNN

#
# ~~~ Make up some fake data
from bnns.data.univar_missing_middle_normalized_12 import x_train, y_train, x_test, y_test

#
# ~~~ Visualize the data
grid = x_test
green_curve = y_test
fig,ax = plt.subplots(figsize=(12,6))
plt.plot( grid, green_curve, "--", color="green", label="Ground Truth" )
plt.plot( x_train, y_train, "o", color="green", label="Training Data" )
plt.ylim([-0.5,0.4])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#
# ~~~ Instantiate an untrained BNN
torch.manual_seed(2024)
BNN = SequentialGaussianBNN(
        nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),    # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
BNN.rho = lambda x:x
with torch.no_grad():
    for p in BNN.model_std.parameters():
        p.data = torch.clamp( p.data, min=1e-6 )

#
# ~~~ Visualize the untrained BNN
def plot_bnn():
    with torch.no_grad():
        predictions = torch.stack([ BNN(grid,resample_weights=True) for _ in range(200) ])
    mean, std = predictions.mean(dim=0).flatten(), predictions.std(dim=0).flatten()
    lo, hi = mean - 2*std, mean + 2*std
    fig,ax = plt.subplots(figsize=(12,6))
    plt.plot( grid, green_curve, "--", color="green", label="Ground Truth" )
    plt.plot( x_train, y_train, "o", color="green", label="Training Data" )
    ax.fill_between( grid, lo, hi, facecolor="blue", interpolate=True, alpha=0.3, label="95% Confidence Interval" )
    plt.plot( grid, mean, color="blue", label="Posaterior Predictive Mean" )
    plt.ylim([-0.5,0.4])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_bnn()

#
# ~~~ Train the BNN
lr = 0.0001
N = 10000
conditional_std = torch.tensor(0.0015)
BNN.conditional_std = conditional_std
likelihood_history = []
kl_history = []
optimizer = optim.Adam( BNN.parameters(), lr=lr )
pbar = trange(N)
for _ in range(N):
    BNN.sample_from_standard_normal()
    kl_div = BNN.weight_kl(exact_formula=True)
    log_lik = BNN.log_likelihood_density(x_train,y_train)
    loss = kl_div - log_lik
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        for p in BNN.model_std.parameters():
            p.data = torch.clamp( p.data, min=1e-6 )
    pbar.set_postfix( {"loss": f"{loss.item():<4.4f}"} )
    _ = pbar.update()
    likelihood_history.append( log_lik.item() )
    kl_history.append( kl_div.item() )

pbar.close()

#
# ~~~ Visualize the trained BNN
plot_bnn()
