from .base import *  # noqa: F403
from .eegnet import EEGNet
from .shallownet import ShallowNet
from .convca import ConvCA
try:
    from .esn import ESN
except ModuleNotFoundError as exc:
    if exc.name != __name__ + ".esn":
        raise
from .trcanet import TRCANet
