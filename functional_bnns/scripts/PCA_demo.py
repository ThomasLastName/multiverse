import numpy as np
from matplotlib import pyplot as plt
from bnns.data.slosh_70_15_15 import out_np


def make_percentage_explained_plot(
    M, title="Percentage of Variance Explained", show=True
):
    m, n = M.shape
    M = M.T if m > n else M
    vals, vecs = np.linalg.eigh(M @ M.T)
    s = np.sqrt(vals[::-1])
    percentage_explained = np.cumsum(s**2) / np.sum(s**2)
    if show:
        plt.plot(percentage_explained)
        plt.xlim([0, 20])
        plt.ylim([0.9, 1.0])
        plt.axhline(0.99, color="red", linestyle="dashed")
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        return s


make_percentage_explained_plot(
    out_np, "Percentage of Variance Explained in PCA on the Raw Data"
)
centered = out_np - np.mean(out_np, axis=0)
make_percentage_explained_plot(
    centered,
    "Percentage of Variance Explained in PCA on the Centered Data (as in the paper)",
)
standardized = (out_np - np.mean(out_np, axis=0)) / (np.std(out_np, axis=0) + 1e-10)
make_percentage_explained_plot(
    centered, "Percentage of Variance Explained in PCA on the Fully Standardized Data"
)
