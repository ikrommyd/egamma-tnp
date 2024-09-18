from __future__ import annotations

import os

import awkward as ak
from coffea.nanoevents import BaseSchema, NanoEventsFactory
from dummy_tag_and_probe_ntuples import mini_tag_and_probe_electrons, mini_tag_and_probe_photons, nano_tag_and_probe_electrons, nano_tag_and_probe_photons

from egamma_tnp import ElectronTagNProbeFromMiniNTuples, ElectronTagNProbeFromNanoNTuples, PhotonTagNProbeFromMiniNTuples, PhotonTagNProbeFromNanoNTuples


def assert_arrays_equal(a1, a2):
    for i in a1.fields:
        assert ak.all(a1[i] == a2[i])
    for j in a2.fields:
        assert ak.all(a1[j] == a2[j])


def test_mini_tag_and_probe_electrons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}

    tag_n_probe = ElectronTagNProbeFromMiniNTuples(
        fileset,
        ["passingCutBasedTight122XV1"],
        cutbased_id="passingCutBasedLoose122XV1",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}, schemaclass=BaseSchema, delayed=False).events()
    solution = mini_tag_and_probe_electrons(events)
    result = tag_n_probe.get_passing_and_failing_probes(
        "passingCutBasedTight122XV1", cut_and_count=False, vars=["el_pt", "el_eta", "truePU", "tag_Ele_eta"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 414
    assert len(result["failing"]) == 113
    assert len(solution[0]) == 414
    assert len(solution[1]) == 113


def test_mini_tag_and_probe_photons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}}}

    tag_n_probe = PhotonTagNProbeFromMiniNTuples(
        fileset,
        ["passingCutBasedTight122XV1"],
        cutbased_id="passingCutBasedLoose122XV1",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}, schemaclass=BaseSchema, delayed=False).events()
    solution = mini_tag_and_probe_photons(events)
    result = tag_n_probe.get_passing_and_failing_probes(
        "passingCutBasedTight122XV1", cut_and_count=False, vars=["ph_et", "ph_eta", "truePU", "tag_Ele_eta"], compute=True
    )["sample"]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 372
    assert len(result["failing"]) == 73
    assert len(solution[0]) == 372
    assert len(solution[1]) == 73


def test_nano_tag_and_probe_electrons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/NanoNTuples_el.root"): "Events"}}}

    tag_n_probe = ElectronTagNProbeFromNanoNTuples(
        fileset,
        ["cutBased >= 4"],
        cutbased_id="cutBased >= 2",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/NanoNTuples_el.root"): "Events"}, schemaclass=BaseSchema, delayed=False).events()
    solution = nano_tag_and_probe_electrons(events)
    result = tag_n_probe.get_passing_and_failing_probes("cutBased >= 4", cut_and_count=False, vars=["el_pt", "el_eta", "PV_npvs", "tag_Ele_eta"], compute=True)[
        "sample"
    ]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 978
    assert len(result["failing"]) == 0
    assert len(solution[0]) == 978
    assert len(solution[1]) == 0


def test_nano_tag_and_probe_photons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/NanoNTuples_ph.root"): "Events"}}}

    tag_n_probe = PhotonTagNProbeFromNanoNTuples(
        fileset,
        ["cutBased >= 3"],
        cutbased_id="cutBased >= 1",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/NanoNTuples_ph.root"): "Events"}, schemaclass=BaseSchema, delayed=False).events()
    solution = nano_tag_and_probe_photons(events)
    result = tag_n_probe.get_passing_and_failing_probes("cutBased >= 3", cut_and_count=False, vars=["ph_pt", "ph_eta", "PV_npvs", "tag_Ele_eta"], compute=True)[
        "sample"
    ]
    assert_arrays_equal(result["passing"], solution[0])
    assert_arrays_equal(result["failing"], solution[1])
    assert len(result["passing"]) == 669
    assert len(result["failing"]) == 135
    assert len(solution[0]) == 669
    assert len(solution[1]) == 135
