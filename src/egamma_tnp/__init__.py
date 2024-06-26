from __future__ import annotations

from egamma_tnp.config import binning
from egamma_tnp.nanoaod_efficiency import ElectronTagNProbeFromNanoAOD, PhotonTagNProbeFromNanoAOD
from egamma_tnp.ntuple_efficiency import ElectronTagNProbeFromNTuples, PhotonTagNProbeFromNTuples

from . import _version

__version__ = _version.__version__
__all__ = ("binning", "ElectronTagNProbeFromNTuples", "ElectronTagNProbeFromNanoAOD", "PhotonTagNProbeFromNTuples", "PhotonTagNProbeFromNanoAOD")


def dir():
    return __all__
