from __future__ import annotations

import os

import dask_awkward as dak
import numpy as np
import uproot

from egamma_tnp.utils import (
    convert_2d_mll_hists_to_1d_hists,
    convert_nd_mll_hists_to_1d_hists,
    create_hists_root_file_for_fitter,
    fill_nd_mll_histograms,
    fill_pt_eta_phi_mll_histograms,
)

fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"}}}


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def test_fitter_histogram_conversion_1d():
    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "pt": passing_probe_evens.el_pt,
            "eta": passing_probe_evens.el_eta,
            "phi": passing_probe_evens.el_phi,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "pt": failing_probe_evens.el_pt,
            "eta": failing_probe_evens.el_eta,
            "phi": failing_probe_evens.el_phi,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("pt_bins", [5, 50, 200, 500])
    egamma_tnp.binning.set("eta_bins", [-2.5, 0, 2.5])
    egamma_tnp.binning.set("phi_bins", [-3.32, 3.32])

    hmll1d = fill_pt_eta_phi_mll_histograms(
        passing_probes,
        failing_probes,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        vars=["pt", "eta", "phi"],
    )

    res1d = convert_2d_mll_hists_to_1d_hists(hmll1d)

    assert list(res1d["pt"]["barrel"]["passing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["pt"]["barrel"]["failing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["pt"]["endcap_loweta"]["passing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["pt"]["endcap_loweta"]["failing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["pt"]["endcap_higheta"]["passing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["pt"]["endcap_higheta"]["failing"].keys()) == ["pt_5p00To50p00", "pt_50p00To200p00", "pt_200p00To500p00"]
    assert list(res1d["eta"]["entire"]["passing"].keys()) == ["eta_m2p50To0p00", "eta_0p00To2p50"]
    assert list(res1d["eta"]["entire"]["failing"].keys()) == ["eta_m2p50To0p00", "eta_0p00To2p50"]
    assert list(res1d["phi"]["entire"]["passing"].keys()) == ["phi_m3p32To3p32"]
    assert list(res1d["phi"]["entire"]["failing"].keys()) == ["phi_m3p32To3p32"]

    assert_histograms_equal(res1d["pt"]["barrel"]["passing"]["pt_5p00To50p00"], hmll1d["pt"]["barrel"]["passing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["barrel"]["failing"]["pt_5p00To50p00"], hmll1d["pt"]["barrel"]["failing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["barrel"]["passing"]["pt_50p00To200p00"], hmll1d["pt"]["barrel"]["passing"][50j:200j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["barrel"]["failing"]["pt_50p00To200p00"], hmll1d["pt"]["barrel"]["failing"][50j:200j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["barrel"]["passing"]["pt_200p00To500p00"], hmll1d["pt"]["barrel"]["passing"][200j:500j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["barrel"]["failing"]["pt_200p00To500p00"], hmll1d["pt"]["barrel"]["failing"][200j:500j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["endcap_loweta"]["passing"]["pt_5p00To50p00"], hmll1d["pt"]["endcap_loweta"]["passing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["endcap_loweta"]["failing"]["pt_5p00To50p00"], hmll1d["pt"]["endcap_loweta"]["failing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["endcap_loweta"]["passing"]["pt_50p00To200p00"], hmll1d["pt"]["endcap_loweta"]["passing"][50j:200j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["endcap_loweta"]["failing"]["pt_50p00To200p00"], hmll1d["pt"]["endcap_loweta"]["failing"][50j:200j:sum, :], flow=False)
    assert_histograms_equal(
        res1d["pt"]["endcap_loweta"]["passing"]["pt_200p00To500p00"], hmll1d["pt"]["endcap_loweta"]["passing"][200j:500j:sum, :], flow=False
    )
    assert_histograms_equal(
        res1d["pt"]["endcap_loweta"]["failing"]["pt_200p00To500p00"], hmll1d["pt"]["endcap_loweta"]["failing"][200j:500j:sum, :], flow=False
    )
    assert_histograms_equal(res1d["pt"]["endcap_higheta"]["passing"]["pt_5p00To50p00"], hmll1d["pt"]["endcap_higheta"]["passing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(res1d["pt"]["endcap_higheta"]["failing"]["pt_5p00To50p00"], hmll1d["pt"]["endcap_higheta"]["failing"][5j:50j:sum, :], flow=False)
    assert_histograms_equal(
        res1d["pt"]["endcap_higheta"]["passing"]["pt_50p00To200p00"], hmll1d["pt"]["endcap_higheta"]["passing"][50j:200j:sum, :], flow=False
    )
    assert_histograms_equal(
        res1d["pt"]["endcap_higheta"]["failing"]["pt_50p00To200p00"], hmll1d["pt"]["endcap_higheta"]["failing"][50j:200j:sum, :], flow=False
    )
    assert_histograms_equal(
        res1d["pt"]["endcap_higheta"]["passing"]["pt_200p00To500p00"], hmll1d["pt"]["endcap_higheta"]["passing"][200j:500j:sum, :], flow=False
    )
    assert_histograms_equal(
        res1d["pt"]["endcap_higheta"]["failing"]["pt_200p00To500p00"], hmll1d["pt"]["endcap_higheta"]["failing"][200j:500j:sum, :], flow=False
    )
    assert_histograms_equal(res1d["eta"]["entire"]["passing"]["eta_m2p50To0p00"], hmll1d["eta"]["entire"]["passing"][-2.5j:0j:sum, :], flow=False)
    assert_histograms_equal(res1d["eta"]["entire"]["failing"]["eta_m2p50To0p00"], hmll1d["eta"]["entire"]["failing"][-2.5j:0j:sum, :], flow=False)
    assert_histograms_equal(res1d["eta"]["entire"]["passing"]["eta_0p00To2p50"], hmll1d["eta"]["entire"]["passing"][0j:2.5j:sum, :], flow=False)
    assert_histograms_equal(res1d["eta"]["entire"]["failing"]["eta_0p00To2p50"], hmll1d["eta"]["entire"]["failing"][0j:2.5j:sum, :], flow=False)
    assert_histograms_equal(res1d["phi"]["entire"]["passing"]["phi_m3p32To3p32"], hmll1d["phi"]["entire"]["passing"][-3.32j:3.32j:sum, :], flow=False)
    assert_histograms_equal(res1d["phi"]["entire"]["failing"]["phi_m3p32To3p32"], hmll1d["phi"]["entire"]["failing"][-3.32j:3.32j:sum, :], flow=False)

    egamma_tnp.binning.reset_all()


def test_fitter_histogram_conversion_3d():
    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "pt": passing_probe_evens.el_pt,
            "eta": passing_probe_evens.el_eta,
            "phi": passing_probe_evens.el_phi,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "pt": failing_probe_evens.el_pt,
            "eta": failing_probe_evens.el_eta,
            "phi": failing_probe_evens.el_phi,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("pt_bins", [5, 50, 200, 500])
    egamma_tnp.binning.set("eta_bins", [-2.5, 0, 2.5])
    egamma_tnp.binning.set("phi_bins", [-3.32, 3.32])

    hmll3d = fill_nd_mll_histograms(
        passing_probes,
        failing_probes,
        vars=["pt", "eta", "phi"],
    )

    res3d, binning = convert_nd_mll_hists_to_1d_hists(hmll3d, axes=["eta", "pt"])
    binning_solution = [
        "eta_m2p50To0p00_pt_5p00To50p00",
        "eta_0p00To2p50_pt_5p00To50p00",
        "eta_m2p50To0p00_pt_50p00To200p00",
        "eta_0p00To2p50_pt_50p00To200p00",
        "eta_m2p50To0p00_pt_200p00To500p00",
        "eta_0p00To2p50_pt_200p00To500p00",
    ]

    assert list(res3d["passing"].keys()) == binning_solution
    assert list(res3d["failing"].keys()) == binning_solution
    assert binning["vars"] == ["eta", "pt"]
    binning_names = [x["name"] for x in binning["bins"]]
    zfill_length = len(str(len(binning_solution)))
    assert binning_names == [f"bin{str(i).zfill(zfill_length)}_{x}" for i, x in enumerate(binning_solution)]

    assert_histograms_equal(res3d["passing"]["eta_m2p50To0p00_pt_5p00To50p00"], hmll3d["passing"][5j:50j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_m2p50To0p00_pt_5p00To50p00"], hmll3d["failing"][5j:50j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["passing"]["eta_0p00To2p50_pt_5p00To50p00"], hmll3d["passing"][5j:50j:sum, 0j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_0p00To2p50_pt_5p00To50p00"], hmll3d["failing"][5j:50j:sum, 0j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["passing"]["eta_m2p50To0p00_pt_50p00To200p00"], hmll3d["passing"][50j:200j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_m2p50To0p00_pt_50p00To200p00"], hmll3d["failing"][50j:200j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["passing"]["eta_0p00To2p50_pt_50p00To200p00"], hmll3d["passing"][50j:200j:sum, 0j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_0p00To2p50_pt_50p00To200p00"], hmll3d["failing"][50j:200j:sum, 0j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["passing"]["eta_m2p50To0p00_pt_200p00To500p00"], hmll3d["passing"][200j:500j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_m2p50To0p00_pt_200p00To500p00"], hmll3d["failing"][200j:500j:sum, -2.5j:0j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["passing"]["eta_0p00To2p50_pt_200p00To500p00"], hmll3d["passing"][200j:500j:sum, 0j:2.5j:sum, sum, :], flow=False)
    assert_histograms_equal(res3d["failing"]["eta_0p00To2p50_pt_200p00To500p00"], hmll3d["failing"][200j:500j:sum, 0j:2.5j:sum, sum, :], flow=False)

    egamma_tnp.binning.reset_all()


def test_fitter_histogram_saving_1d():
    import pickle

    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "pt": passing_probe_evens.el_pt,
            "eta": passing_probe_evens.el_eta,
            "phi": passing_probe_evens.el_phi,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "pt": failing_probe_evens.el_pt,
            "eta": failing_probe_evens.el_eta,
            "phi": failing_probe_evens.el_phi,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("pt_bins", [5, 50, 200, 500])
    egamma_tnp.binning.set("eta_bins", [-2.5, 0, 2.5])
    egamma_tnp.binning.set("phi_bins", [-3.32, 3.32])

    hmll1d = fill_pt_eta_phi_mll_histograms(
        passing_probes,
        failing_probes,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        vars=["pt", "eta", "phi"],
    )

    create_hists_root_file_for_fitter(hmll1d, "1d_hists.root", "1d_binning.pkl")

    for region in ["barrel", "endcap_loweta", "endcap_higheta"]:
        with uproot.open(f"1d_hists_pt_{region}.root") as f:
            binning = pickle.load(open(f"1d_binning_pt_{region}.pkl", "rb"))
            for bin in binning["bins"]:
                name = bin["name"]
                min_pt = bin["vars"]["pt"]["min"] * 1j
                max_pt = bin["vars"]["pt"]["max"] * 1j

                saved_passing_hist = f[f"{name}_Pass"].to_hist()
                saved_failing_hist = f[f"{name}_Fail"].to_hist()

                assert_histograms_equal(saved_passing_hist, hmll1d["pt"][region]["passing"][min_pt:max_pt:sum, :], flow=False)
                assert_histograms_equal(saved_failing_hist, hmll1d["pt"][region]["failing"][min_pt:max_pt:sum, :], flow=False)

    with uproot.open("1d_hists_eta_entire.root") as f:
        binning = pickle.load(open("1d_binning_eta_entire.pkl", "rb"))
        for bin in binning["bins"]:
            name = bin["name"]
            min_eta = bin["vars"]["eta"]["min"] * 1j
            max_eta = bin["vars"]["eta"]["max"] * 1j

            saved_passing_hist = f[f"{name}_Pass"].to_hist()
            saved_failing_hist = f[f"{name}_Fail"].to_hist()

            assert_histograms_equal(saved_passing_hist, hmll1d["eta"]["entire"]["passing"][min_eta:max_eta:sum, :], flow=False)
            assert_histograms_equal(saved_failing_hist, hmll1d["eta"]["entire"]["failing"][min_eta:max_eta:sum, :], flow=False)

    with uproot.open("1d_hists_phi_entire.root") as f:
        binning = pickle.load(open("1d_binning_phi_entire.pkl", "rb"))
        for bin in binning["bins"]:
            name = bin["name"]
            min_phi = bin["vars"]["phi"]["min"] * 1j
            max_phi = bin["vars"]["phi"]["max"] * 1j

            saved_passing_hist = f[f"{name}_Pass"].to_hist()
            saved_failing_hist = f[f"{name}_Fail"].to_hist()

            assert_histograms_equal(saved_passing_hist, hmll1d["phi"]["entire"]["passing"][min_phi:max_phi:sum, :], flow=False)
            assert_histograms_equal(saved_failing_hist, hmll1d["phi"]["entire"]["failing"][min_phi:max_phi:sum, :], flow=False)

    os.remove("1d_hists_pt_barrel.root")
    os.remove("1d_hists_pt_endcap_loweta.root")
    os.remove("1d_hists_pt_endcap_higheta.root")
    os.remove("1d_hists_eta_entire.root")
    os.remove("1d_hists_phi_entire.root")
    os.remove("1d_binning_pt_barrel.pkl")
    os.remove("1d_binning_pt_endcap_loweta.pkl")
    os.remove("1d_binning_pt_endcap_higheta.pkl")
    os.remove("1d_binning_eta_entire.pkl")
    os.remove("1d_binning_phi_entire.pkl")

    egamma_tnp.binning.reset_all()


def test_fitter_histogram_saving_3d():
    import pickle

    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "pt": passing_probe_evens.el_pt,
            "eta": passing_probe_evens.el_eta,
            "phi": passing_probe_evens.el_phi,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "pt": failing_probe_evens.el_pt,
            "eta": failing_probe_evens.el_eta,
            "phi": failing_probe_evens.el_phi,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("pt_bins", [5, 50, 200, 500])
    egamma_tnp.binning.set("eta_bins", [-2.5, 0, 2.5])
    egamma_tnp.binning.set("phi_bins", [-3.32, 3.32])

    hmll3d = fill_nd_mll_histograms(
        passing_probes,
        failing_probes,
        vars=["pt", "eta", "phi"],
    )

    create_hists_root_file_for_fitter(hmll3d, "3d_hists.root", "3d_binning.pkl", axes=["pt", "eta"])

    with uproot.open("3d_hists.root") as f:
        binning = pickle.load(open("3d_binning.pkl", "rb"))
        for bin in binning["bins"]:
            name = bin["name"]
            min_eta = bin["vars"]["eta"]["min"] * 1j
            max_eta = bin["vars"]["eta"]["max"] * 1j
            min_pt = bin["vars"]["pt"]["min"] * 1j
            max_pt = bin["vars"]["pt"]["max"] * 1j

            saved_passing_hist = f[f"{name}_Pass"].to_hist()
            saved_failing_hist = f[f"{name}_Fail"].to_hist()

            assert_histograms_equal(saved_passing_hist, hmll3d["passing"][min_pt:max_pt:sum, min_eta:max_eta:sum, sum, :], flow=False)
            assert_histograms_equal(saved_failing_hist, hmll3d["failing"][min_pt:max_pt:sum, min_eta:max_eta:sum, sum, :], flow=False)

    os.remove("3d_hists.root")
    os.remove("3d_binning.pkl")

    egamma_tnp.binning.reset_all()


def test_fitter_histogram_saving_against_reference():
    import pickle

    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "el_pt": passing_probe_evens.el_pt,
            "el_sc_eta": passing_probe_evens.el_sc_eta,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "el_pt": failing_probe_evens.el_pt,
            "el_sc_eta": failing_probe_evens.el_sc_eta,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("el_pt_bins", [10, 20, 35, 50, 100, 200, 500])
    egamma_tnp.binning.set("el_sc_eta_bins", [-2.5, -2.0, -1.566, -1.4442, -0.8, 0.0, 0.8, 1.4442, 1.566, 2.0, 2.5])

    hmll3d = fill_nd_mll_histograms(
        passing_probes,
        failing_probes,
        vars=["el_sc_eta", "el_pt"],
    )

    create_hists_root_file_for_fitter(hmll3d, "3d_hists.root", "3d_binning.pkl", axes=["el_sc_eta", "el_pt"])

    binning = pickle.load(open("3d_binning.pkl", "rb"))
    modified_binning = binning.copy()

    for bin in modified_binning["bins"]:
        current_cut = bin["cut"]
        new_cut = "tag_Ele_pt > 35 && abs(tag_sc_eta) < 2.17 && el_q*tag_Ele_q < 0 && " + current_cut + " && tag_Ele_trigMVA > 0.92  "
        new_cut = "tag_Ele_pt > 35 && abs(tag_sc_eta) < 2.17 && el_q*tag_Ele_q < 0 && " + current_cut
        bin["cut"] = new_cut

    for bin in modified_binning["bins"][:10]:
        current_cut = bin["cut"]
        new_cut = current_cut + " && tag_Ele_trigMVA > 0.92  "
        bin["cut"] = new_cut

    egm_tnp_analysis_binning = pickle.load(open(os.path.abspath("tests/samples/fitter_binning.pkl"), "rb"))
    assert modified_binning == egm_tnp_analysis_binning

    with uproot.open("3d_hists.root") as f:
        for bin in binning["bins"]:
            name = bin["name"]
            min_eta = bin["vars"]["el_sc_eta"]["min"] * 1j
            max_eta = bin["vars"]["el_sc_eta"]["max"] * 1j
            min_pt = bin["vars"]["el_pt"]["min"] * 1j
            max_pt = bin["vars"]["el_pt"]["max"] * 1j

            saved_passing_hist = f[f"{name}_Pass"].to_hist()
            saved_failing_hist = f[f"{name}_Fail"].to_hist()

            assert_histograms_equal(saved_passing_hist, hmll3d["passing"][min_eta:max_eta:sum, min_pt:max_pt:sum, :], flow=False)
            assert_histograms_equal(saved_failing_hist, hmll3d["failing"][min_eta:max_eta:sum, min_pt:max_pt:sum, :], flow=False)

    os.remove("3d_hists.root")
    os.remove("3d_binning.pkl")

    egamma_tnp.binning.reset_all()


def test_fitter_histogram_conversion_binning():
    import pickle

    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "el_pt": passing_probe_evens.el_pt,
            "el_sc_eta": passing_probe_evens.el_sc_eta,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "el_pt": failing_probe_evens.el_pt,
            "el_sc_eta": failing_probe_evens.el_sc_eta,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.binning.set("el_pt_bins", [10, 20, 35, 50, 100, 200, 500])
    egamma_tnp.binning.set("el_sc_eta_bins", [-2.5, -2.0, -1.566, -1.4442, -0.8, 0.0, 0.8, 1.4442, 1.566, 2.0, 2.5])

    hmll3d = fill_nd_mll_histograms(
        passing_probes,
        failing_probes,
        vars=["el_sc_eta", "el_pt"],
    )

    res3d, binning = convert_nd_mll_hists_to_1d_hists(hmll3d, axes=["el_sc_eta", "el_pt"])

    modified_binning = binning.copy()

    for bin in modified_binning["bins"]:
        current_cut = bin["cut"]
        new_cut = "tag_Ele_pt > 35 && abs(tag_sc_eta) < 2.17 && el_q*tag_Ele_q < 0 && " + current_cut + " && tag_Ele_trigMVA > 0.92  "
        new_cut = "tag_Ele_pt > 35 && abs(tag_sc_eta) < 2.17 && el_q*tag_Ele_q < 0 && " + current_cut
        bin["cut"] = new_cut

    for bin in modified_binning["bins"][:10]:
        current_cut = bin["cut"]
        new_cut = current_cut + " && tag_Ele_trigMVA > 0.92  "
        bin["cut"] = new_cut

    egm_tnp_analysis_binning = pickle.load(open(os.path.abspath("tests/samples/fitter_binning.pkl"), "rb"))
    assert modified_binning == egm_tnp_analysis_binning

    for bin in binning["bins"]:
        name = bin["name"].split("_", 1)[1]
        min_eta = bin["vars"]["el_sc_eta"]["min"] * 1j
        max_eta = bin["vars"]["el_sc_eta"]["max"] * 1j
        min_pt = bin["vars"]["el_pt"]["min"] * 1j
        max_pt = bin["vars"]["el_pt"]["max"] * 1j

        assert_histograms_equal(res3d["passing"][name], hmll3d["passing"][min_eta:max_eta:sum, min_pt:max_pt:sum, :], flow=False)
        assert_histograms_equal(res3d["failing"][name], hmll3d["failing"][min_eta:max_eta:sum, min_pt:max_pt:sum, :], flow=False)

    egamma_tnp.binning.reset_all()
