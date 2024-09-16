from __future__ import annotations

import os

import pytest
from coffea.dataset_tools import preprocess

from egamma_tnp.triggers import ElePt_WPTight_Gsf


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_pileup_ntuples(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree",
                    os.path.abspath("tests/samples/not_a_file.root"): "fitter_tree",
                },
                "metadata": {"isMC": True, "pileupJSON": os.path.abspath("tests/samples/test_pu_correction.json")},
            }
        }
    else:
        fileset = {
            "sample": {
                "files": {os.path.abspath("tests/samples/TnPNTuples_el.root"): "fitter_tree"},
                "metadata": {"isMC": True, "pileupJSON": os.path.abspath("tests/samples/test_pu_correction.json")},
            }
        }

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(fileset, skip_bad_files=True)
            fileset = fileset_available

    tag_n_probe = ElePt_WPTight_Gsf(
        fileset,
        trigger_pt=32,
        mode="from_mini_ntuples",
        tags_pt_cut=35,
        probes_pt_cut=5,
        use_sc_eta=False,
        avoid_ecal_transition_tags=False,
    )

    res = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "passHltEle32WPTightGsf",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=True,
    )

    if allow_read_errors_with_report:
        histograms = res[0]["sample"]
        report = res[1]["sample"]
        if not do_preprocess:
            assert report.exception[1] == "FileNotFoundError"
    else:
        histograms = res["sample"]

    hpt_pass_barrel, hpt_fail_barrel = histograms["pt"]["barrel"].values()
    hpt_pass_endcap, hpt_fail_endcap = histograms["pt"]["endcap"].values()
    heta_pass, heta_fail = histograms["eta"]["entire"].values()
    hphi_pass, hphi_fail = histograms["phi"]["entire"].values()

    assert hpt_pass_barrel.sum(flow=True).value + hpt_pass_endcap.sum(flow=True).value == 349.0 * 3
    assert hpt_fail_barrel.sum(flow=True).value + hpt_fail_endcap.sum(flow=True).value == (490.0 - 349.0) * 3
    assert heta_pass.sum(flow=True).value == 361.0 * 3
    assert heta_fail.sum(flow=True).value == (505.0 - 361.0) * 3
    assert hphi_pass.sum(flow=True).value == 361.0 * 3
    assert hphi_fail.sum(flow=True).value == (505.0 - 361.0) * 3

    assert hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0] == 0.0
    assert hpt_fail_barrel.values(flow=True)[0] + hpt_fail_endcap.values(flow=True)[0] == 0.0
    assert heta_pass.values(flow=True)[0] == 0.0
    assert heta_fail.values(flow=True)[0] == 0.0
    assert hphi_pass.values(flow=True)[0] == 0.0
    assert hphi_fail.values(flow=True)[0] == 0.0


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_pileup_nanoaod(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/DYto2E.root"): "Events",
                    os.path.abspath("tests/samples/not_a_file.root"): "Events",
                },
                "metadata": {"isMC": True, "pileupJSON": os.path.abspath("tests/samples/test_pu_correction.json")},
            }
        }
    else:
        fileset = {
            "sample": {
                "files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"},
                "metadata": {"isMC": True, "pileupJSON": os.path.abspath("tests/samples/test_pu_correction.json")},
            }
        }

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(fileset, skip_bad_files=True)
            fileset = fileset_available

    tag_n_probe = ElePt_WPTight_Gsf(
        fileset,
        32,
        probes_pt_cut=29,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )

    res = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
        "HLT_Ele32_WPTight_Gsf",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=True,
    )

    if allow_read_errors_with_report:
        histograms = res[0]["sample"]
        report = res[1]["sample"]
        if not do_preprocess:
            assert report.exception[1] == "FileNotFoundError"
    else:
        histograms = res["sample"]

    hpt_pass_barrel, hpt_fail_barrel = histograms["pt"]["barrel"].values()
    hpt_pass_endcap, hpt_fail_endcap = histograms["pt"]["endcap"].values()
    heta_pass, heta_fail = histograms["eta"]["entire"].values()
    hphi_pass, hphi_fail = histograms["phi"]["entire"].values()

    assert hpt_pass_barrel.sum(flow=True).value + hpt_pass_endcap.sum(flow=True).value == 954.0 * 3
    assert hpt_fail_barrel.sum(flow=True).value + hpt_fail_endcap.sum(flow=True).value == (1153.0 - 954.0) * 3
    assert heta_pass.sum(flow=True).value == 954.0 * 3
    assert heta_fail.sum(flow=True).value == (1153.0 - 954.0) * 3
    assert hphi_pass.sum(flow=True).value == 954.0 * 3
    assert hphi_fail.sum(flow=True).value == (1153.0 - 954.0) * 3

    assert hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0] == 0.0
    assert hpt_fail_barrel.values(flow=True)[0] + hpt_fail_endcap.values(flow=True)[0] == 0.0
    assert heta_pass.values(flow=True)[0] == 0.0
    assert heta_fail.values(flow=True)[0] == 0.0
    assert hphi_pass.values(flow=True)[0] == 0.0
    assert hphi_fail.values(flow=True)[0] == 0.0
