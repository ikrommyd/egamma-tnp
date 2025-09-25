from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

from egamma_tnp.config import binning
from egamma_tnp.nanoaod_efficiency import ElectronTagNProbeFromNanoAOD, PhotonTagNProbeFromNanoAOD
from egamma_tnp.nanoaod_sas_ntuples import ScaleAndSmearingNTuplesFromNanoAOD
from egamma_tnp.ntuple_efficiency import (
    ElectronTagNProbeFromMiniNTuples,
    ElectronTagNProbeFromNanoNTuples,
    PhotonTagNProbeFromMiniNTuples,
    PhotonTagNProbeFromNanoNTuples,
)

from . import _version

warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
__version__ = _version.__version__
__all__ = (
    "ElectronTagNProbeFromMiniNTuples",
    "ElectronTagNProbeFromNanoAOD",
    "ElectronTagNProbeFromNanoNTuples",
    "PhotonTagNProbeFromMiniNTuples",
    "PhotonTagNProbeFromNanoAOD",
    "PhotonTagNProbeFromNanoNTuples",
    "ScaleAndSmearingNTuplesFromNanoAOD",
    "binning",
)


def dir():
    return __all__
