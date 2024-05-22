from __future__ import annotations

import os

import numpy as np

from egamma_tnp import ElectronTagNProbeFromNanoAOD

fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def test_histogramming_default_vars():
    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        True,
        filter="Ele30",
        filterbit=1,
        tags_pt_cut=30,
        use_sc_eta=True,
        tags_abseta_cut=2.50,
        probes_pt_cut=27,
        trigger_pt=30,
    )
    hcnc1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
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
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        compute=True,
    )["sample"]
    hcnc3d = tag_n_probe.get_nd_tnp_histograms(cut_and_count=True, compute=True)["sample"]
    hmll3d = tag_n_probe.get_nd_tnp_histograms(cut_and_count=False, compute=True)["sample"]

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


def test_histogramming_custom_vars():
    import egamma_tnp

    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        True,
        filter="Ele30",
        filterbit=1,
        tags_pt_cut=30,
        use_sc_eta=True,
        tags_abseta_cut=2.50,
        probes_pt_cut=27,
        trigger_pt=30,
    )

    egamma_tnp.config.set("el_r9_bins", np.linspace(0.1, 1.05, 100).tolist())

    hmll1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        compute=True,
    )["sample"]

    hmll3d = tag_n_probe.get_nd_tnp_histograms(cut_and_count=False, vars=["el_eta", "el_r9"], compute=True)["sample"]

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, :], flow=False)

    egamma_tnp.config.reset_all()


def test_histogramming_non_probe_vars():
    import egamma_tnp

    tag_n_probe = ElectronTagNProbeFromNanoAOD(
        fileset,
        True,
        filter="Ele30",
        filterbit=1,
        tags_pt_cut=30,
        use_sc_eta=True,
        tags_abseta_cut=2.50,
        probes_pt_cut=27,
        trigger_pt=30,
    )

    egamma_tnp.config.set("MET_pt_bins", np.linspace(0, 200, 10).tolist())
    egamma_tnp.config.set("luminosityBlock_bins", np.linspace(0, 1000, 11).tolist())
    egamma_tnp.config.set("tag_Ele_pt_bins", egamma_tnp.config.get("pt_bins"))

    hmll1d = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        cut_and_count=False,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        compute=True,
    )["sample"]

    hmll3d = tag_n_probe.get_nd_tnp_histograms(cut_and_count=False, vars=["el_eta", "tag_Ele_pt", "MET_pt", "luminosityBlock"], compute=True)["sample"]

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, sum, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, sum, sum, :], flow=False)

    egamma_tnp.config.reset_all()
