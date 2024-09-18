from __future__ import annotations

import os

import awkward as ak
import pytest
from coffea.nanoevents import NanoEventsFactory
from dummy_tag_and_probe_nanoaod import tag_and_probe_electrons, tag_and_probe_photons

from egamma_tnp import ElectronTagNProbeFromNanoAOD, PhotonTagNProbeFromNanoAOD


def assert_arrays_equal(a1, a2):
    for i in a1.fields:
        assert ak.all(a1[i] == a2[i])
    for j in a2.fields:
        assert ak.all(a1[j] == a2[j])


def test_tag_and_probe_electrons_trigger():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        ["HLT_Ele32_WPTight_Gsf"],
        trigger_pt=[32],
        filterbit=[1],
        cutbased_id="cutBased >= 4",
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=True,
        require_event_to_pass_hlt_filter=True,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_electrons(events, is_id=False)
    result = tag_n_probe.get_passing_and_failing_probes(
        "HLT_Ele32_WPTight_Gsf", cut_and_count=False, vars=["Electron_pt", "tag_Ele_eta", "el_pt", "el_eta", "MET_pt", "event"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 467
    assert len(result["failing"]) == 183
    assert len(solution[0]) == 467
    assert len(solution[1]) == 183


def test_tag_and_probe_electrons_id():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        ["cutBased >= 4"],
        trigger_pt=[32],
        filterbit=[1],
        cutbased_id=None,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=True,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_electrons(events, is_id=True)
    result = tag_n_probe.get_passing_and_failing_probes(
        "cutBased >= 4", cut_and_count=False, vars=["Electron_pt", "tag_Ele_eta", "el_pt", "el_eta", "MET_pt", "event"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 649
    assert len(result["failing"]) == 0
    assert len(solution[0]) == 649
    assert len(solution[1]) == 0


@pytest.mark.parametrize("start_from_diphotons", [True, False])
def test_tag_and_probe_photons_trigger(start_from_diphotons):
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = PhotonTagNProbeFromNanoAOD(
        fileset,
        ["HLT_Ele32_WPTight_Gsf"],
        is_electron_filter=[True],
        start_from_diphotons=start_from_diphotons,
        cutbased_id="cutBased >= 3",
        trigger_pt=[32],
        filterbit=[1],
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=False,
        require_event_to_pass_hlt_filter=True,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_photons(events, start_from_diphotons, is_id=False)
    result = tag_n_probe.get_passing_and_failing_probes(
        "HLT_Ele32_WPTight_Gsf", cut_and_count=False, vars=["Photon_pt", "tag_Ele_eta", "ph_pt", "ph_eta", "MET_pt", "event"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    if start_from_diphotons:
        assert len(result["passing"]) == 336
        assert len(result["failing"]) == 101
        assert len(solution[0]) == 336
        assert len(solution[1]) == 101
    else:
        assert len(result["passing"]) == 441
        assert len(result["failing"]) == 122
        assert len(solution[0]) == 441
        assert len(solution[1]) == 122


@pytest.mark.parametrize("start_from_diphotons", [True, False])
def test_tag_and_probe_photons_id(start_from_diphotons):
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    tag_n_probe = PhotonTagNProbeFromNanoAOD(
        fileset,
        ["cutBased >= 3"],
        is_electron_filter=[True],
        start_from_diphotons=start_from_diphotons,
        cutbased_id=None,
        trigger_pt=[32],
        filterbit=[1],
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        use_sc_eta=False,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/DYto2E.root"): "Events"}, delayed=False).events()
    solution = tag_and_probe_photons(events, start_from_diphotons, is_id=True)
    result = tag_n_probe.get_passing_and_failing_probes(
        "cutBased >= 3", cut_and_count=False, vars=["Photon_pt", "tag_Ele_eta", "ph_pt", "ph_eta", "MET_pt", "event"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    if start_from_diphotons:
        assert len(result["passing"]) == 436
        assert len(result["failing"]) == 146
        assert len(solution[0]) == 436
        assert len(solution[1]) == 146
    else:
        assert len(result["passing"]) == 562
        assert len(result["failing"]) == 170
        assert len(solution[0]) == 562
        assert len(solution[1]) == 170
