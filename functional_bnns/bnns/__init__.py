#
# ~~~ See distribution
from quality_of_life.my_base_utils import my_warn
from pkg_resources import get_distribution
dist = get_distribution("bnns")

#
# ~~~ Fetch local package version
__version__ = dist.version
from .NoPriorBNNs import *
from .WeightPriorBNNs import *
from .GPPriorBNNs import *

#
# ~~~ Deprecation warnings
class MixtureWeightPrior2015BNN(MixturePrior2015BNN):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs )
        my_warn("MixtureWeightPrior2015BNN has been renamed to MixturePrior2015BNN. The old naming is deprecated and may be removed in a future release.")

class IndepLocScaleSequentialBNN(IndepLocScaleBNN):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs )
        my_warn("IndepLocScaleSequentialBNN has been renamed to IndepLocScaleBNN. The old naming is deprecated and may be removed in a future release.")

class SequentialGaussianBNN(GaussianBNN):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        my_warn("SequentialGaussianBNN has been renamed to GaussianBNN. The old naming is deprecated and may be removed in a future release.")
