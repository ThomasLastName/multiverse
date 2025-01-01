#
# ~~~ See distribution
from pkg_resources import get_distribution
dist = get_distribution('bnns')

#
# ~~~ Fetch local package version
__version__ = dist.version

#
# ~~~ Fetch some of the main classes
from bnns.SequentialGaussianBNN import SequentialGaussianBNN

