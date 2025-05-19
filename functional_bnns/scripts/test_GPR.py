
import torch
from bnns.GPR import GPY
from bnns.GPR import simple_mean_zero_RPF_kernel_GP as GP
from matplotlib import pyplot as plt
from tqdm import trange
from quality_of_life.my_base_utils import support_for_progress_bars

#
# ~~~ Make up data
torch.manual_seed(2025)
x_train = torch.linspace(-1,1,5).reshape(-1,1)
y_train = x_train.abs() + 0.05*torch.randn_like(x_train)
x_test = torch.linspace(-1,1,301).reshape(-1,1)
y_test = x_test.abs()

#
# ~~~ Do GPR
fig, axs = plt.subplots(1,2,figsize=(12,6))
with support_for_progress_bars():
    for j,(ax,model) in enumerate(zip( axs, (GP,GPY) )):
        message = f"GPR {'WITHOUT ' if model==GP else ''}Using GPyTorch"
        for _ in trange( 20, desc=message ):
            #
            # ~~~ Define the model
            try:    gp = model( out_features=1, etas=[0.0001] )
            except: gp = model( out_features=1, eta=0.0001 )
            #
            # ~~~ Fit the model and get predictions
            gp.fit( x_train, y_train )
            mu_post, sigma_post = gp.post_mu_and_Sigma(x_test)
            predictions = gp( x_test, n=100 )
        #
        # ~~~ Plot the results
        ax.plot( x_test, mu_post, color="blue" )
        for p in predictions: ax.plot( x_test, p, color="blue", alpha=0.1 )
        ax.plot( x_test, y_test, linestyle="--", color="green" )
        ax.scatter( x_train, y_train, color="green" )
        ax.set_title(message)
        ax.grid()

plt.tight_layout()
plt.show()
