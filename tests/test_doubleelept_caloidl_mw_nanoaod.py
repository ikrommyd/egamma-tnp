from __future__ import annotations

import os

import pytest
from coffea.dataset_tools import preprocess

from egamma_tnp.triggers import (
    DoubleElePt_CaloIdL_MW_SeededLeg,
    DoubleElePt_CaloIdL_MW_UnseededLeg,
)


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_without_compute(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/DYto2E.root"): "Events",
                    os.path.abspath("tests/samples/not_a_file.root"): "Events",
                }
            }
        }
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(fileset, skip_bad_files=True)
            fileset = fileset_available

    tag_n_probe_leg1 = DoubleElePt_CaloIdL_MW_SeededLeg(
        fileset,
        33,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )
    tag_n_probe_leg2 = DoubleElePt_CaloIdL_MW_UnseededLeg(
        fileset,
        33,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )

    for tag_n_probe in [tag_n_probe_leg1, tag_n_probe_leg2]:
        res = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
            "HLT_DoubleEle33_CaloIdL_MW_SeededLeg" if tag_n_probe is tag_n_probe_leg1 else "HLT_DoubleEle33_CaloIdL_MW_UnseededLeg",
            uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
            compute=False,
            scheduler=None,
            progress=False,
        )

        if allow_read_errors_with_report:
            histograms = res[0]["sample"]
        else:
            histograms = res["sample"]

        hpt_pass_barrel, hpt_fail_barrel = histograms["pt"]["barrel"].values()
        hpt_pass_endcap, hpt_fail_endcap = histograms["pt"]["endcap"].values()
        heta_pass, heta_fail = histograms["eta"]["entire"].values()
        hphi_pass, hphi_fail = histograms["phi"]["entire"].values()

        assert hpt_pass_barrel.sum(flow=True).value + hpt_pass_endcap.sum(flow=True).value == 0.0
        assert hpt_fail_barrel.sum(flow=True).value + hpt_fail_endcap.sum(flow=True).value == 0.0
        assert heta_pass.sum(flow=True).value == 0.0
        assert heta_fail.sum(flow=True).value == 0.0
        assert hphi_pass.sum(flow=True).value == 0.0
        assert hphi_fail.sum(flow=True).value == 0.0

        assert hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0] == 0.0
        assert hpt_fail_barrel.values(flow=True)[0] + hpt_fail_endcap.values(flow=True)[0] == 0.0
        assert heta_pass.values(flow=True)[0] == 0.0
        assert heta_fail.values(flow=True)[0] == 0.0
        assert hphi_pass.values(flow=True)[0] == 0.0
        assert hphi_fail.values(flow=True)[0] == 0.0


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_local_compute(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/DYto2E.root"): "Events",
                    os.path.abspath("tests/samples/not_a_file.root"): "Events",
                }
            }
        }
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(fileset, skip_bad_files=True)
            fileset = fileset_available

    tag_n_probe_leg1 = DoubleElePt_CaloIdL_MW_SeededLeg(
        fileset,
        33,
        tags_pt_cut=35,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )
    tag_n_probe_leg2 = DoubleElePt_CaloIdL_MW_UnseededLeg(
        fileset,
        33,
        tags_pt_cut=35,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )

    for tag_n_probe, target_pt, target_eta_phi, total_sum in zip(
        [tag_n_probe_leg1, tag_n_probe_leg2],
        [0.0, 0.0],
        [0.0, 0.0],
        [893.0, 893.0],
    ):
        res = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
            "HLT_DoubleEle33_CaloIdL_MW_SeededLeg" if tag_n_probe is tag_n_probe_leg1 else "HLT_DoubleEle33_CaloIdL_MW_UnseededLeg",
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

        assert hpt_pass_barrel.sum(flow=True).value + hpt_pass_endcap.sum(flow=True).value == target_pt
        assert hpt_fail_barrel.sum(flow=True).value + hpt_fail_endcap.sum(flow=True).value == total_sum - target_pt
        assert heta_pass.sum(flow=True).value == target_eta_phi
        assert heta_fail.sum(flow=True).value == total_sum - target_eta_phi
        assert hphi_pass.sum(flow=True).value == target_eta_phi
        assert hphi_fail.sum(flow=True).value == total_sum - target_eta_phi

        assert hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0] == 0.0
        assert hpt_fail_barrel.values(flow=True)[0] + hpt_fail_endcap.values(flow=True)[0] == 0.0
        assert heta_pass.values(flow=True)[0] == 0.0
        assert heta_fail.values(flow=True)[0] == 0.0
        assert hphi_pass.values(flow=True)[0] == 0.0
        assert hphi_fail.values(flow=True)[0] == 0.0


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_distributed_compute(do_preprocess, allow_read_errors_with_report):
    from distributed import Client

    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/DYto2E.root"): "Events",
                    os.path.abspath("tests/samples/not_a_file.root"): "Events",
                }
            }
        }
    else:
        fileset = {"sample": {"files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}}}

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(fileset, skip_bad_files=True)
            fileset = fileset_available

    tag_n_probe_leg1 = DoubleElePt_CaloIdL_MW_SeededLeg(
        fileset,
        33,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )
    tag_n_probe_leg2 = DoubleElePt_CaloIdL_MW_UnseededLeg(
        fileset,
        33,
        probes_pt_cut=33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
    )

    with Client():
        for tag_n_probe, target_pt, target_eta_phi, total_sum in zip(
            [tag_n_probe_leg1, tag_n_probe_leg2],
            [0.0, 0.0],
            [0.0, 0.0],
            [893.0, 893.0],
        ):
            res = tag_n_probe.get_1d_pt_eta_phi_tnp_histograms(
                "HLT_DoubleEle33_CaloIdL_MW_SeededLeg" if tag_n_probe is tag_n_probe_leg1 else "HLT_DoubleEle33_CaloIdL_MW_UnseededLeg",
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

            assert hpt_pass_barrel.sum(flow=True).value + hpt_pass_endcap.sum(flow=True).value == target_pt
            assert hpt_fail_barrel.sum(flow=True).value + hpt_fail_endcap.sum(flow=True).value == total_sum - target_pt
            assert heta_pass.sum(flow=True).value == target_eta_phi
            assert heta_fail.sum(flow=True).value == total_sum - target_eta_phi
            assert hphi_pass.sum(flow=True).value == target_eta_phi
            assert hphi_fail.sum(flow=True).value == total_sum - target_eta_phi

            assert hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0] == 0.0
            assert hpt_fail_barrel.values(flow=True)[0] + hpt_fail_endcap.values(flow=True)[0] == 0.0
            assert heta_pass.values(flow=True)[0] == 0.0
            assert heta_fail.values(flow=True)[0] == 0.0
            assert hphi_pass.values(flow=True)[0] == 0.0
            assert hphi_fail.values(flow=True)[0] == 0.0
