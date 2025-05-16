
LR = 0.0005
N_EPOCHS = 20000
BATCH_SIZE = 300

# Provided...
import math
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from torch import nn
log_phi = lambda eps: -eps**2/(2*0.0001) - math.log(math.sqrt(2*math.pi*0.0001))  # N( 0, .01**2 )
from bnns import GPPriorBNN as BNN  # includes prior, Z, G_\theta, and the means to estimate KL-div
from bnns.data.univar_missing_middle_normalized_12 import D_train, x_val, y_val  # x_1,...,x_m, y_1,...,y_m
torch.manual_seed(2025)
f = nn.Sequential( nn.Linear(1,100), nn.ReLU(), nn.Linear(100,1) )  # f(x;w)

# Code...
device = "cuda" if torch.cuda.is_available() else "cpu"
x_val, y_val = x_val.to(device).reshape(-1,1), y_val.to(device)
bnn = BNN(*f).to(device)                # initilaize theta
optimizer = torch.optim.Adam( bnn.parameters(), lr=LR )         # hyper-parameters
dataloader = torch.utils.data.DataLoader( D_train, batch_size=BATCH_SIZE )  # hyper-parameters
pbar = tqdm( total=N_EPOCHS )
train_loss = []
val_acc = []
for epoch in range(N_EPOCHS):                                          # hyper-parameters
    for (X,y) in dataloader:
        X, y = X.reshape(-1,1).to(device), y.to(device)
        bnn.resample_weights()          # draw a new sample z_sampled
        yhat = bnn(X)                   # automatically constrains theta, and sets w_sampled
        log_lik = log_phi(y-yhat).sum()
        kl_div  = bnn.functional_kl()   # also samples new measuement points (seenext section)
        vi_loss = kl_div - log_lik
        vi_loss.backward()              # compute the gradient of vi_loss using auto-diff
        optimizer.step()                # take a step using ADAM
    if epoch%50==0:
        with torch.no_grad(): preds = bnn(x_val,n=100)
        predictive_median = preds.median(dim=0).values
        rmse_of_predictive_median = torch.mean( (y_val-predictive_median)**2 ).sqrt().item()
        pbar.set_postfix({ "train_loss":f"{vi_loss.item():4.4}", "val_acc": f"{rmse_of_predictive_median:4.4}" })
        train_loss.append( vi_loss.item() )
        val_acc.append( rmse_of_predictive_median )
    _ = pbar.update()

# Validate results
pbar.close()
x_val, y_val = x_val.reshape(-1,1).to(device), y_val.to(device)
predictive_median = preds.median(dim=0).values
rmse_of_predictive_median = torch.mean( (y_val-predictive_median)**2 ).sqrt().item()
print(f"Test accuracy is {rmse_of_predictive_median}")

grid = torch.linspace(-1,1,301).to("cuda").reshape(-1,1)
preds = bnn(grid,n=100).detach()
for j in range(100): plt.plot( grid.cpu(), preds[j].cpu(), color="blue", alpha=0.05 )

plt.scatter( x_val.cpu(), y_val.cpu(), s=10, c="k" )
plt.ylim(-.5,.5)
plt.show()
