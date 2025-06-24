# Improved implementation avoiding recursive function calls

import torch  # we're not using autograd or anything; numpy would also work
from tqdm import trange
from matplotlib import pyplot as plt
from quality_of_life.my_visualization_utils import GifMaker

make_gif = True
plot_g = True

x_train = torch.linspace(0, 2 * torch.pi, 1001)  # `grid` for grid-search optimization
y_train = 1 - torch.cos(2 * x_train)  # == f(x_train)
q = torch.where(
    (0 <= x_train) & (x_train <= 2 * torch.pi), 1 / (2 * torch.pi), torch.tensor(0.0)
)  # uniform
wt = -1

# Do Frank-Wolfe
gif = GifMaker()
for t in trange(500):
    # Set the learning rate
    alpha = 2 / (t + 2)
    # Take the graident (only evaluated at x_train, rather than a proper function)
    g = torch.log(q) + 1 - torch.log(y_train)
    # Compute w_t
    wt = x_train[g.argmin()]
    # Define u_t as a piece-wise linear "hat function" centered around w_t
    r = (
        alpha ** (1 / 5) + 0.0001
    )  # take hat functions with smaller supports during later iterations
    u = (1 / r) * torch.maximum(
        torch.tensor(0.0),
        torch.minimum(
            (x_train - (wt - r)) / (wt - (wt - r)), 1 - (x_train - wt) / ((wt + r) - wt)
        ),
    )
    if make_gif:
        if plot_g:
            _ = plt.plot(
                x_train, g, "--", color="orange"
            )  # true posterior density function
            _ = plt.title(f"The function g_{t} which needs to be minimized")
            _ = plt.xlim([-0.01, 2 * torch.pi + 0.01])
        else:
            _ = plt.plot(
                x_train,
                y_train / (2 * torch.pi),
                "--",
                color="green",
                label="True Post.",
            )  # true posterior density function
            _ = plt.plot(
                x_train, q, "-", color="blue", label="Approx. Post."
            )  # approximate posterior density function
            _ = plt.axvline(wt, color="red", linestyle="dashed", label=f"Loc. of w_{t}")
            _ = plt.xlim([-0.01, 2 * torch.pi + 0.01])
            _ = plt.ylim([0, 1 / 3])
            _ = plt.legend(loc="upper left")
        gif.capture()
    # Define q_{t+1} (only evaluated at x_train, rather than a proper function)
    q = (1 - alpha) * q + alpha * u

_ = plt.plot(
    x_train,
    y_train / (2 * torch.pi),
    "--",
    color="green",
    label="True Posterior Density",
)  # true posterior density function
_ = plt.plot(
    x_train, q, "-", color="blue", label="Approximate Posterior Density"
)  # approximate posterior density function
_ = plt.xlim([0, 2 * torch.pi])
_ = plt.ylim([0, 1 / 3])
_ = plt.suptitle(
    f"Estimated Integral is {(y_train/q).median()}. True integral is {2*torch.pi}."
)
_ = plt.legend(loc="upper left")
if make_gif:
    gif.capture()
    gif.develop(destination="Tom's Idea (plot of g)" if plot_g else "Tom's Idea", fps=1)
else:
    plt.show()
