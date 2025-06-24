try:
    from bnns.utils.handling import my_warn
except:
    from warnings import warn as my_warn

#
# ~~~ https://chatgpt.com/share/68586694-1abc-8001-92ac-3db42eef50f6
try:
    from pkg_resources import get_distribution

    __version__ = get_distribution("bnns").version
except Exception as e:
    try:
        from importlib.metadata import version as importlib_version
    except ImportError:
        from importlib_metadata import (
            version as importlib_version,
        )  # ~~~ for Python <3.8, reportedly
    try:
        __version__ = importlib_version("bnns")
        my_warn(
            f"Could not retrieve version via pkg_resources due to a possible dependency conflict: {e}"
        )
    except Exception as e2:
        my_warn(
            f"Could not determine bnns version via pkg_resources or importlib.\n"
            f"pkg_resources error: {e}\n"
            f"importlib error: {e2}"
        )
        __version__ = "unknown"

#
# ~~~ Fetch local package version
from .NoPriorBNNs import *
from .WeightPriorBNNs import *
from .GPPriorBNNs import *
from .Ensemble import *


#
# ~~~ Deprecation warnings
class MixtureWeightPrior2015BNN(MixturePrior2015BNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        my_warn(
            "MixtureWeightPrior2015BNN has been renamed to MixturePrior2015BNN. The old naming is deprecated and may be removed in a future release."
        )


class IndepLocScaleSequentialBNN(IndepLocScaleBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        my_warn(
            "IndepLocScaleSequentialBNN has been renamed to IndepLocScaleBNN. The old naming is deprecated and may be removed in a future release."
        )


class SequentialGaussianBNN(GaussianBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        my_warn(
            "SequentialGaussianBNN has been renamed to GaussianBNN. The old naming is deprecated and may be removed in a future release."
        )
