from __future__ import annotations

import os

import numpy as np
import pytest

from egamma_tnp import ElectronTagNProbeFromMiniNTuples, PhotonTagNProbeFromMiniNTuples


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


@pytest.mark.parametrize("tag_n_probe_class", [ElectronTagNProbeFromMiniNTuples, PhotonTagNProbeFromMiniNTuples])
def test_histogramming_default_vars(tag_n_probe_class):
    if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passHltEle30WPTightGsf"],
            cutbased_id="passingCutBasedTight122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passingCutBasedTight122XV1"],
            cutbased_id="passingCutBasedLoose122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )

    hcnc1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=True,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        compute=True,
    )["sample"]
    hmll1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        compute=True,
    )["sample"]
    hcnc3d = tag_n_probe.get_nd_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=True,
        compute=True,
    )["sample"]
    hmll3d = tag_n_probe.get_nd_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=False,
        compute=True,
    )["sample"]

    assert_histograms_equal(hcnc1d["pt"]["barrel"]["passing"], hcnc3d["passing"][:, -1.4442j:1.4442j:sum, sum], flow=False)
    assert_histograms_equal(hcnc1d["pt"]["barrel"]["failing"], hcnc3d["failing"][:, -1.4442j:1.4442j:sum, sum], flow=False)
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_loweta"]["passing"], hcnc3d["passing"][:, -2j:-1.566j:sum, sum] + hcnc3d["passing"][:, 1.566j:2j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_loweta"]["failing"], hcnc3d["failing"][:, -2j:-1.566j:sum, sum] + hcnc3d["failing"][:, 1.566j:2j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_higheta"]["passing"], hcnc3d["passing"][:, -2.5j:-2j:sum, sum] + hcnc3d["passing"][:, 2j:2.5j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_higheta"]["failing"], hcnc3d["failing"][:, -2.5j:-2j:sum, sum] + hcnc3d["failing"][:, 2j:2.5j:sum, sum], flow=False
    )
    assert_histograms_equal(hcnc1d["eta"]["entire"]["passing"], hcnc3d["passing"][35j::sum, -2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hcnc1d["eta"]["entire"]["failing"], hcnc3d["failing"][35j::sum, -2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hcnc1d["phi"]["entire"]["passing"], hcnc3d["passing"][35j::sum, -2.5j:2.5j:sum, :], flow=False)
    assert_histograms_equal(hcnc1d["phi"]["entire"]["failing"], hcnc3d["failing"][35j::sum, -2.5j:2.5j:sum, :], flow=False)

    assert_histograms_equal(hmll1d["pt"]["barrel"]["passing"], hmll3d["passing"][:, -1.4442j:1.4442j:sum, sum, :], flow=False)
    assert_histograms_equal(hmll1d["pt"]["barrel"]["failing"], hmll3d["failing"][:, -1.4442j:1.4442j:sum, sum, :], flow=False)
    assert_histograms_equal(
        hmll1d["pt"]["endcap_loweta"]["passing"], hmll3d["passing"][:, -2j:-1.566j:sum, sum, :] + hmll3d["passing"][:, 1.566j:2j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_loweta"]["failing"], hmll3d["failing"][:, -2j:-1.566j:sum, sum, :] + hmll3d["failing"][:, 1.566j:2j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_higheta"]["passing"], hmll3d["passing"][:, -2.5j:-2j:sum, sum, :] + hmll3d["passing"][:, 2j:2.5j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_higheta"]["failing"], hmll3d["failing"][:, -2.5j:-2j:sum, sum, :] + hmll3d["failing"][:, 2j:2.5j:sum, sum, :], flow=False
    )
    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][35j::sum, -2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][35j::sum, -2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["phi"]["entire"]["passing"], hmll3d["passing"][35j::sum, -2.5j:2.5j:sum, :, :], flow=False)
    assert_histograms_equal(hmll1d["phi"]["entire"]["failing"], hmll3d["failing"][35j::sum, -2.5j:2.5j:sum, :, :], flow=False)


@pytest.mark.parametrize("tag_n_probe_class", [ElectronTagNProbeFromMiniNTuples, PhotonTagNProbeFromMiniNTuples])
def test_histogramming_custom_vars(tag_n_probe_class):
    import egamma_tnp

    if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passHltEle30WPTightGsf"],
            cutbased_id="passingCutBasedTight122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passingCutBasedTight122XV1"],
            cutbased_id="passingCutBasedLoose122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )

    egamma_tnp.binning.set("el_r9_bins", np.linspace(0.1, 1.05, 100).tolist())
    egamma_tnp.binning.set("ph_r9_bins", np.linspace(0.1, 1.05, 100).tolist())

    hmll1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        compute=True,
    )["sample"]

    if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples:
        hmll3d = tag_n_probe.get_nd_tnp_histograms(
            "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
            cut_and_count=False,
            vars=["el_eta", "el_r9"],
            compute=True,
        )["sample"]
    else:
        hmll3d = tag_n_probe.get_nd_tnp_histograms(
            "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
            cut_and_count=False,
            vars=["ph_eta", "ph_r9"],
            compute=True,
        )["sample"]

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, :], flow=False)

    egamma_tnp.binning.reset_all()


@pytest.mark.parametrize("tag_n_probe_class", [ElectronTagNProbeFromMiniNTuples, PhotonTagNProbeFromMiniNTuples])
def test_histogramming_non_probe_vars(tag_n_probe_class):
    import egamma_tnp

    if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passHltEle30WPTightGsf"],
            cutbased_id="passingCutBasedTight122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_ph.root"): "fitter_tree"}}}
        tag_n_probe = tag_n_probe_class(
            fileset,
            ["passingCutBasedTight122XV1"],
            cutbased_id="passingCutBasedLoose122XV1",
            use_sc_eta=True,
            tags_pt_cut=30,
        )

    egamma_tnp.binning.set("tag_sc_eta_bins", egamma_tnp.binning.get("eta_bins"))
    egamma_tnp.binning.set("lumi_bins", np.linspace(0, 1000, 11).tolist())

    hmll1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        compute=True,
    )["sample"]

    if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples:
        hmll3d = tag_n_probe.get_nd_tnp_histograms(
            "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
            cut_and_count=False,
            vars=["el_eta", "tag_sc_eta", "lumi"],
            compute=True,
        )["sample"]
    else:
        hmll3d = tag_n_probe.get_nd_tnp_histograms(
            "passHltEle30WPTightGsf" if tag_n_probe_class == ElectronTagNProbeFromMiniNTuples else "passingCutBasedTight122XV1",
            cut_and_count=False,
            vars=["ph_eta", "tag_sc_eta", "lumi"],
            compute=True,
        )["sample"]

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, -2.5j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, -2.5j:2.5j:sum, sum, :], flow=False)

    egamma_tnp.binning.reset_all()
