from __future__ import annotations

import os

import awkward as ak
from coffea.nanoevents import NanoEventsFactory
from dummy_sas_ntuplizer import sas_ntuples

from egamma_tnp import ScaleAndSmearingNTuplesFromNanoAOD


def assert_arrays_equal(a1, a2):
    assert sorted(a1.fields) == sorted(a2.fields)
    for i in a1.fields:
        assert ak.all(a1[i] == a2[i])


def test_sas_ntuplizer():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    ntuplizer = ScaleAndSmearingNTuplesFromNanoAOD(
        fileset,
        lead_pt_cut=23.0,
        sublead_pt_cut=12.0,
        eta_cut=2.17,
        trigger_paths="Ele23_Ele12_CaloIdL_TrackIdL_IsoVL*",
        avoid_ecal_transition=True,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, mode="virtual").events()
    solution = sas_ntuples(events)
    vars = {
        "Photon": [
            "pt",
            "eta",
            "phi",
            "seedGain",
            "superclusterEta",
            "r9",
            "sieie",
            "hoe",
            "hoe_PUcorr",
            "s4",
            "phiWidth",
            "etaWidth",
            "sieip",
            "sipip",
            "esEffSigmaRR",
            "esEnergyOverRawE",
            "cutBased",
            "mvaID",
            "mvaID_WP80",
            "mvaID_WP90",
        ],
        "Electron": [
            "pt",
            "eta",
            "phi",
            "seedGain",
            "charge",
            "superclusterEta",
            "r9",
            "sieie",
            "hoe",
            "convVeto",
            "eInvMinusPInv",
            "fbrem",
            "cutBased",
            "cutBased_HEEP",
            "mvaIso",
            "mvaIso_WP80",
            "mvaIso_WP90",
            "mvaIso_WPHZZ",
            "mvaNoIso",
            "mvaNoIso_WP80",
            "mvaNoIso_WP90",
            "pfRelIso03_all",
            "pfRelIso03_chg",
            "dr03EcalRecHitSumEt",
            "dr03HcalDepth1TowerSumEt",
            "dr03TkSumPt",
        ],
        "PuppiMET": ["pt", "phi", "sumEt"],
        "Rho": "all",
    }
    result = ntuplizer.get_ntuples(mass_range=(60, 120), vars=vars, flat=True, compute=True)["sample"]
    assert_arrays_equal(result, solution)
    assert len(result) == 686
    assert len(solution) == 686
