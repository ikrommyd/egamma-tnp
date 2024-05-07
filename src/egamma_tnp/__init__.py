from egamma_tnp.config import config
from egamma_tnp.nanoaod_efficiency import ElectronTagNProbeFromNanoAOD
from egamma_tnp.ntuple_efficiency import ElectronTagNProbeFromNTuples

from . import _version

__version__ = _version.__version__
__all__ = ("config", "ElectronTagNProbeFromNTuples", "ElectronTagNProbeFromNanoAOD")


def dir():
    return __all__
