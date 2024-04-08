from egamma_tnp.config import config
from egamma_tnp.nanoaod_efficiency import TagNProbeFromNanoAOD
from egamma_tnp.ntuple_efficiency import TagNProbeFromNTuples

from . import _version

__version__ = _version.__version__
__all__ = ("config", "TagNProbeFromNTuples", "TagNProbeFromNanoAOD")
