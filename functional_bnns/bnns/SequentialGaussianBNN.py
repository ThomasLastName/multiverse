
from bnns.WeightPriorBNNs import SequentialGaussianBNN as SameClassNewLocation
from quality_of_life.my_base_utils import my_warn

class SequentialGaussianBNN(SameClassNewLocation):
    def __init__(self,*args,**kwargs):
        my_warn("Deprecation warning: `bnns.SequentialGaussianBNN.SequentialGaussianBNN` has been relocated to `bnns.WeightPriorBNN.SequentialGaussianBNN`.")
        super().__init__(*args,**kwargs)