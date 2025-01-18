#
# ~~~ See distribution
from pkg_resources import get_distribution
dist = get_distribution('bnns')

#
# ~~~ Fetch local package version
__version__ = dist.version
from .NoPriorBNNs import *
from .WeightPriorBNNs import *
from .GPPriorBNNs import *
