from bnns.WeightPriorBNNs import GaussianBNN as SameClassNewLocation
from bnns.utils.handling import my_warn

my_warn(
    "Deprecation warning: `SequentialGaussianBNN` has been renamed to `GaussianBNN` and relocated to `WeightPriorBNN.py` as of Jan/2025. This import will be killed in the future and will no longer work."
)


class SequentialGaussianBNN(SameClassNewLocation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
