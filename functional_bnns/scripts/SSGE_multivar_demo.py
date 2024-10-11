
import torch
from bnns.SSGE import SpectralSteinEstimator
from matplotlib import pyplot as plt
import math
import numpy as np
from quality_of_life.my_base_utils import support_for_progress_bars
from quality_of_life.my_visualization_utils import *
from tqdm import trange

torch.manual_seed(1234)
M = 1000   # ~~~ will be implicity rounded *down* the the nearest square number: int(sqrt(M))**2
J = 20
eta = 0.0095
make_gif = True
D = 3       # ~~~ in the final implemtnation of functional BNN's, this is the size of the training dataset
n_test = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

test_points = torch.randn( n_test, D, device=device )
ground_truth = lambda x: -x
score = ground_truth(test_points)



def make_hist(m,lim=1000000):
    samples = torch.randn( m, D, device=device )
    try:
        score_estimator = SpectralSteinEstimator( samples=samples, eta=eta, J=J )
    except RuntimeError:
        score_estimator = SpectralSteinEstimator( samples=samples, eta=eta, J=J, h=False )
    est_score = score_estimator(test_points)
    errors = ((est_score-score)**2).sum(dim=-1).cpu()
    lim = min(np.percentile(errors,90),lim)
    errors = errors[errors<np.percentile(errors,90)]
    plt.hist( errors, bins=50 )
    plt.xlim( left=0, right=lim )
    plt.title(f"SSGE Errors at a Scattering of Points in R^{D} when M={m}")
    plt.tight_layout()
    plt.grid()
    return np.percentile(errors,90)

if make_gif:
    gif = GifMaker()
    lim = 100000000
    with support_for_progress_bars():
        for m in trange(0,M,5):
            new_lim = make_hist( m+5, lim )
            lim = min(lim,new_lim)
            gif.capture()
    # gif.develop( destination="intended_filename", fps=24 ) # if you want to save the .gif
else:
    make_hist(M)
    plt.show()




# def error(xy):
#     xy = torch.from_numpy(xy).to(torch.get_default_dtype())
#     pred = score_estimator(xy,samples)
#     target = ground_truth(xy)
#     return ((pred-target)**2).sum(dim=-1)

# func_surf( x=test_points[:,0], y=test_points[:,1], f=error )
