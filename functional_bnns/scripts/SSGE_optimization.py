
import time
import torch
from bnns.SSGE import SpectralSteinEstimator as SSGE

M = 3000
d = 100
samples = torch.randn( M, d, device = "cuda" if torch.cuda.is_available() else "cpu" )

#
# ~~~ Time the new method
tick = time.time()
ssge = SSGE( samples, eta=0.5, J=100 )
tock = time.time()
new_time = tock-tick
print(f"Time using new method: {new_time:.4f} seconds")

#
# ~~~ Time the old method
tick = time.time()
ssge_old = SSGE( samples, eta=0.5, J=100, old=True )
tock = time.time()
old_time = tock-tick

print(f"Time using old method: {old_time:.4f} seconds")

#
# ~~~ Compare
assert torch.allclose( ssge.avg_jac, ssge_old.avg_jac )
print(f"The results are identical, but the new method is: {old_time / new_time:.2f}x as fast.")

