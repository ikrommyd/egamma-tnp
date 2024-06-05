from __future__ import annotations

import os

import pytest
from coffea.nanoevents import NanoEventsFactory
from dask_awkward.lib.testutils import assert_eq
from dummy_tag_and_probe_nanoaod import tag_and_probe_electrons, tag_and_probe_photons

from egamma_tnp import ElectronTagNProbeFromNanoAOD, PhotonTagNProbeFromNanoAOD


def test_tag_and_probe_electrons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        True,
        filter="Ele32",
        trigger_pt=32,
        filterbit=1,
        cutbased_id="cutBased >= 4",
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=True,
        hlt_filter="Ele32_WPTight_Gsf",
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_electrons(events)
    result = tag_n_probe.get_tnp_arrays(cut_and_count=False, vars=["Electron_pt", "tag_Ele_eta", "el_pt", "el_eta", "MET_pt", "event"], compute=True)["sample"]
    assert_eq(result["passing"], solution[0])
    assert_eq(result["failing"], solution[1])


@pytest.mark.parametrize("start_from_diphotons", [True, False])
def test_tag_and_probe_photons(start_from_diphotons):
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = PhotonTagNProbeFromNanoAOD(
        fileset,
        True,
        filter="Ele32",
        is_electron_filter=True,
        start_from_diphotons=start_from_diphotons,
        cutbased_id="cutBased >= 3",
        trigger_pt=32,
        filterbit=1,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=False,
        hlt_filter="Ele32_WPTight_Gsf",
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_photons(events, start_from_diphotons)
    result = tag_n_probe.get_tnp_arrays(cut_and_count=False, vars=["Photon_pt", "tag_Ele_eta", "ph_pt", "ph_eta", "MET_pt", "event"], compute=True)["sample"]
    assert_eq(result["passing"], solution[0])
    assert_eq(result["failing"], solution[1])
