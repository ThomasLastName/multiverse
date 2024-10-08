#
# ~~~ See distribution
from pkg_resources import get_distribution, DistributionNotFound
dist = get_distribution('bnns')

#
# ~~~ Fetch local package version
__version__ = dist.version
