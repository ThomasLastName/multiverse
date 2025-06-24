import torch
from bnns.SSGE import SpectralSteinEstimator as SSGE
from matplotlib import pyplot as plt
from quality_of_life import my_plt_utils as mpu
from tqdm import tqdm


bw = 0.2
eta = 0.001
how_many_to_plot = 5
sample_sizes_to_test = [(j + 1) * 100 for j in range(100)]
input_grid = torch.linspace(-1, 1, 301)


def plot(predictions, fig, ax):
    for j, p in enumerate(predictions.T):
        ax.plot(input_grid, p, label=f"{j+1}")
    ax.legend()
    plt.grid()
    plt.tight_layout()
    return fig, ax


gif = mpu.GifMaker()

fig, ax = plt.subplots(figsize=(10, 5))
for n in tqdm(sample_sizes_to_test):
    samples = torch.rand(n, 1) * 2 - 1
    ssge = SSGE(samples, J=how_many_to_plot, sigma=bw, eta=eta)
    predictions = ssge.Phi(input_grid)
    fix, ax = plot(predictions, fig, ax)
    gif.capture()
