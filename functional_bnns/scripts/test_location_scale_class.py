import math
import torch
from bnns.utils import LocationScaleLogDensity


def f(z):
    return -(z**2) / 2 - math.log(math.sqrt(2 * torch.pi))


log_gaussian_density = LocationScaleLogDensity(f)
