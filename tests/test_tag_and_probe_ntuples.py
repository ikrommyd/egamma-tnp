from __future__ import annotations

import os

from coffea.nanoevents import BaseSchema, NanoEventsFactory
from dask_awkward.lib.testutils import assert_eq
from dummy_tag_and_probe_ntuples import tag_and_probe_electrons, tag_and_probe_photons

from egamma_tnp import ElectronTagNProbeFromNTuples, PhotonTagNProbeFromNTuples


def test_tag_and_probe_electrons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}

    tag_n_probe = ElectronTagNProbeFromNTuples(
        fileset,
        "passingCutBasedTight122XV1",
        cutbased_id="passingCutBasedLoose122XV1",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}, schemaclass=BaseSchema, delayed=False).events()
    solution = tag_and_probe_electrons(events)
    result = tag_n_probe.get_tnp_arrays(cut_and_count=False, vars=["el_pt", "el_eta", "truePU", "tag_Ele_eta"], compute=True)["sample"]
    assert_eq(result["passing"], solution[0])
    assert_eq(result["failing"], solution[1])
    assert len(result["passing"]) == 414
    assert len(result["failing"]) == 113
    assert len(solution[0]) == 414
    assert len(solution[1]) == 113


def test_tag_and_probe_photons():
    fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}}}

    tag_n_probe = PhotonTagNProbeFromNTuples(
        fileset,
        "passingCutBasedTight122XV1",
        cutbased_id="passingCutBasedLoose122XV1",
        use_sc_eta=True,
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
    )

    events = NanoEventsFactory.from_root({os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}, schemaclass=BaseSchema, delayed=False).events()
    solution = tag_and_probe_photons(events)
    result = tag_n_probe.get_tnp_arrays(cut_and_count=False, vars=["ph_et", "ph_eta", "truePU", "tag_Ele_eta"], compute=True)["sample"]
    assert_eq(result["passing"], solution[0])
    assert_eq(result["failing"], solution[1])
    assert len(result["passing"]) == 372
    assert len(result["failing"]) == 73
    assert len(solution[0]) == 372
    assert len(solution[1]) == 73
