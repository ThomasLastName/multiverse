#
# ~~~ See distribution
from pkg_resources import get_distribution, DistributionNotFound
dist = get_distribution('bnns')

#
# ~~~ Fetch local package version
__version__ = dist.version

#
# ~~~ Fatch package location on machine
import os
__path__ = os.path.dirname(os.path.abspath(__file__))
