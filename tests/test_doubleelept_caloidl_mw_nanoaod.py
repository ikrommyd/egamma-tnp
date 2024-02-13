import os

import numpy as np
import pytest
from coffea.dataset_tools import preprocess

from egamma_tnp.triggers import DoubleElePt_CaloIdL_MW


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
        fileset = {
            "sample": {
                "files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}
            }
        }

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(
                fileset, skip_bad_files=True
            )
            fileset = fileset_available

    tag_n_probe = DoubleElePt_CaloIdL_MW(
        fileset,
        33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
    )

    res_leg1 = tag_n_probe.get_tnp_histograms(
        leg="first",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=False,
        scheduler=None,
        progress=False,
    )
    res_leg2 = tag_n_probe.get_tnp_histograms(
        leg="second",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=False,
        scheduler=None,
        progress=False,
    )
    res_both = tag_n_probe.get_tnp_histograms(
        leg="both",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=False,
        scheduler=None,
        progress=False,
    )

    if allow_read_errors_with_report:
        histograms_leg1 = res_leg1[0]["sample"]
        histograms_leg2 = res_leg2[0]["sample"]
        histograms_both = res_both[0]["sample"]
    else:
        histograms_leg1 = res_leg1["sample"]
        histograms_leg2 = res_leg2["sample"]
        histograms_both = res_both["sample"]

    hpt_pass_barrel_leg1, hpt_all_barrel_leg1 = histograms_leg1["leg1"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_leg1, hpt_all_endcap_leg1 = histograms_leg1["leg1"]["pt"][
        "endcap"
    ].values()
    heta_pass_leg1, heta_all_leg1 = histograms_leg1["leg1"]["eta"]["entire"].values()
    hphi_pass_leg1, hphi_all_leg1 = histograms_leg1["leg1"]["phi"]["entire"].values()
    hpt_pass_barrel_leg2, hpt_all_barrel_leg2 = histograms_leg2["leg2"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_leg2, hpt_all_endcap_leg2 = histograms_leg2["leg2"]["pt"][
        "endcap"
    ].values()
    heta_pass_leg2, heta_all_leg2 = histograms_leg2["leg2"]["eta"]["entire"].values()
    hphi_pass_leg2, hphi_all_leg2 = histograms_leg2["leg2"]["phi"]["entire"].values()
    hpt_pass_barrel_both_leg1, hpt_all_barrel_both_leg1 = histograms_both["leg1"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_both_leg1, hpt_all_endcap_both_leg1 = histograms_both["leg1"]["pt"][
        "endcap"
    ].values()
    heta_pass_both_leg1, heta_all_both_leg1 = histograms_both["leg1"]["eta"][
        "entire"
    ].values()
    hphi_pass_both_leg1, hphi_all_both_leg1 = histograms_both["leg1"]["phi"][
        "entire"
    ].values()
    hpt_pass_barrel_both_leg2, hpt_all_barrel_both_leg2 = histograms_both["leg2"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_both_leg2, hpt_all_endcap_both_leg2 = histograms_both["leg2"]["pt"][
        "endcap"
    ].values()
    heta_pass_both_leg2, heta_all_both_leg2 = histograms_both["leg2"]["eta"][
        "entire"
    ].values()
    hphi_pass_both_leg2, hphi_all_both_leg2 = histograms_both["leg2"]["phi"][
        "entire"
    ].values()

    assert np.all(
        hpt_pass_barrel_leg1.values(flow=True)
        == hpt_pass_barrel_both_leg1.values(flow=True)
    )
    assert np.all(
        hpt_pass_endcap_leg1.values(flow=True)
        == hpt_pass_endcap_both_leg1.values(flow=True)
    )
    assert np.all(
        heta_pass_leg1.values(flow=True) == heta_pass_both_leg1.values(flow=True)
    )
    assert np.all(
        hphi_pass_leg1.values(flow=True) == hphi_pass_both_leg1.values(flow=True)
    )
    assert np.all(
        hpt_pass_barrel_leg2.values(flow=True)
        == hpt_pass_barrel_both_leg2.values(flow=True)
    )
    assert np.all(
        hpt_pass_endcap_leg2.values(flow=True)
        == hpt_pass_endcap_both_leg2.values(flow=True)
    )
    assert np.all(
        heta_pass_leg2.values(flow=True) == heta_pass_both_leg2.values(flow=True)
    )
    assert np.all(
        hphi_pass_leg2.values(flow=True) == hphi_pass_both_leg2.values(flow=True)
    )

    assert hpt_pass_barrel_leg1.sum(flow=True) == hpt_pass_barrel_both_leg1.sum(
        flow=True
    )
    assert hpt_pass_endcap_leg1.sum(flow=True) == hpt_pass_endcap_both_leg1.sum(
        flow=True
    )
    assert heta_pass_leg1.sum(flow=True) == heta_pass_both_leg1.sum(flow=True)
    assert hphi_pass_leg1.sum(flow=True) == hphi_pass_both_leg1.sum(flow=True)
    assert hpt_pass_barrel_leg2.sum(flow=True) == hpt_pass_barrel_both_leg2.sum(
        flow=True
    )
    assert hpt_pass_endcap_leg2.sum(flow=True) == hpt_pass_endcap_both_leg2.sum(
        flow=True
    )
    assert heta_pass_leg2.sum(flow=True) == heta_pass_both_leg2.sum(flow=True)
    assert hphi_pass_leg2.sum(flow=True) == hphi_pass_both_leg2.sum(flow=True)

    assert (
        hpt_pass_barrel_leg1.sum(flow=True) + hpt_pass_endcap_leg1.sum(flow=True) == 0.0
    )
    assert (
        hpt_all_barrel_leg1.sum(flow=True) + hpt_all_endcap_leg1.sum(flow=True) == 0.0
    )
    assert heta_pass_leg1.sum(flow=True) == 0.0
    assert heta_all_leg1.sum(flow=True) == 0.0
    assert hphi_pass_leg1.sum(flow=True) == 0.0
    assert hphi_all_leg1.sum(flow=True) == 0.0
    assert (
        hpt_pass_barrel_leg2.sum(flow=True) + hpt_pass_endcap_leg2.sum(flow=True) == 0.0
    )
    assert (
        hpt_all_barrel_leg2.sum(flow=True) + hpt_all_endcap_leg2.sum(flow=True) == 0.0
    )
    assert heta_pass_leg2.sum(flow=True) == 0.0
    assert heta_all_leg2.sum(flow=True) == 0.0
    assert hphi_pass_leg2.sum(flow=True) == 0.0
    assert hphi_all_leg2.sum(flow=True) == 0.0

    assert (
        hpt_pass_barrel_leg1.values(flow=True)[0]
        + hpt_pass_endcap_leg1.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel_leg1.values(flow=True)[0]
        + hpt_all_endcap_leg1.values(flow=True)[0]
        == 0.0
    )
    assert heta_pass_leg1.values(flow=True)[0] == 0.0
    assert heta_all_leg1.values(flow=True)[0] == 0.0
    assert hphi_pass_leg1.values(flow=True)[0] == 0.0
    assert hphi_all_leg1.values(flow=True)[0] == 0.0
    assert (
        hpt_pass_barrel_leg2.values(flow=True)[0]
        + hpt_pass_endcap_leg2.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel_leg2.values(flow=True)[0]
        + hpt_all_endcap_leg2.values(flow=True)[0]
        == 0.0
    )
    assert heta_pass_leg2.values(flow=True)[0] == 0.0
    assert heta_all_leg2.values(flow=True)[0] == 0.0
    assert hphi_pass_leg2.values(flow=True)[0] == 0.0
    assert hphi_all_leg2.values(flow=True)[0] == 0.0


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
        fileset = {
            "sample": {
                "files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}
            }
        }

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(
                fileset, skip_bad_files=True
            )
            fileset = fileset_available

    tag_n_probe = DoubleElePt_CaloIdL_MW(
        fileset,
        33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
    )

    res_leg1 = tag_n_probe.get_tnp_arrays(
        leg="first",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    res_leg2 = tag_n_probe.get_tnp_arrays(
        leg="second",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    res_both = tag_n_probe.get_tnp_arrays(
        leg="both",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    if allow_read_errors_with_report:
        arrays_leg1 = res_leg1[0]["sample"]
        report_arrays_leg1 = res_leg1[1]["sample"]
        arrays_leg2 = res_leg2[0]["sample"]
        report_arrays_leg2 = res_leg2[1]["sample"]
        arrays_both = res_both[0]["sample"]
        report_arrays_both = res_both[1]["sample"]
        if not do_preprocess:
            assert report_arrays_leg1.exception[1] == "FileNotFoundError"
            assert report_arrays_leg2.exception[1] == "FileNotFoundError"
            assert report_arrays_both.exception[1] == "FileNotFoundError"
    else:
        arrays_leg1 = res_leg1["sample"]
        arrays_leg2 = res_leg2["sample"]
        arrays_both = res_both["sample"]

    res_leg1 = tag_n_probe.get_tnp_histograms(
        leg="first",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    res_leg2 = tag_n_probe.get_tnp_histograms(
        leg="second",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    res_both = tag_n_probe.get_tnp_histograms(
        leg="both",
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
        compute=True,
        scheduler=None,
        progress=False,
    )
    if allow_read_errors_with_report:
        histograms_leg1 = res_leg1[0]["sample"]
        histograms_leg2 = res_leg2[0]["sample"]
        histograms_both = res_both[0]["sample"]
    else:
        histograms_leg1 = res_leg1["sample"]
        histograms_leg2 = res_leg2["sample"]
        histograms_both = res_both["sample"]
    if allow_read_errors_with_report:
        histograms_leg1 = res_leg1[0]["sample"]
        report_histograms_leg1 = res_leg1[1]["sample"]
        histograms_leg2 = res_leg2[0]["sample"]
        report_histograms_leg2 = res_leg2[1]["sample"]
        histograms_both = res_both[0]["sample"]
        report_histograms_both = res_both[1]["sample"]
        if not do_preprocess:
            assert report_histograms_leg1.exception[1] == "FileNotFoundError"
            assert report_histograms_leg2.exception[1] == "FileNotFoundError"
            assert report_histograms_both.exception[1] == "FileNotFoundError"
    else:
        histograms_leg1 = res_leg1["sample"]
        histograms_leg2 = res_leg2["sample"]
        histograms_both = res_both["sample"]

    arrays_pass_leg1, arrays_all_leg1 = arrays_leg1["leg1"]
    arrays_pass_leg2, arrays_all_leg2 = arrays_leg2["leg2"]
    arrays_pass_both_leg1, arrays_all_both_leg1 = arrays_both["leg1"]
    arrays_pass_both_leg2, arrays_all_both_leg2 = arrays_both["leg2"]

    for field in ["pt", "eta", "phi"]:
        assert np.all(arrays_pass_leg1[field] == arrays_pass_both_leg1[field])
        assert np.all(arrays_pass_leg2[field] == arrays_pass_both_leg2[field])
        assert np.all(arrays_all_leg1[field] == arrays_all_both_leg1[field])
        assert np.all(arrays_all_leg2[field] == arrays_all_both_leg2[field])

    if not preprocess:
        assert report_arrays_leg1.exception[1] == "FileNotFoundError"
        assert report_arrays_leg2.exception[1] == "FileNotFoundError"
        assert report_arrays_both.exception[1] == "FileNotFoundError"
        assert report_histograms_leg1.exception[1] == "FileNotFoundError"
        assert report_histograms_leg2.exception[1] == "FileNotFoundError"
        assert report_histograms_both.exception[1] == "FileNotFoundError"

    hpt_pass_barrel_leg1, hpt_all_barrel_leg1 = histograms_leg1["leg1"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_leg1, hpt_all_endcap_leg1 = histograms_leg1["leg1"]["pt"][
        "endcap"
    ].values()
    heta_pass_leg1, heta_all_leg1 = histograms_leg1["leg1"]["eta"]["entire"].values()
    hphi_pass_leg1, hphi_all_leg1 = histograms_leg1["leg1"]["phi"]["entire"].values()
    hpt_pass_barrel_leg2, hpt_all_barrel_leg2 = histograms_leg2["leg2"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_leg2, hpt_all_endcap_leg2 = histograms_leg2["leg2"]["pt"][
        "endcap"
    ].values()
    heta_pass_leg2, heta_all_leg2 = histograms_leg2["leg2"]["eta"]["entire"].values()
    hphi_pass_leg2, hphi_all_leg2 = histograms_leg2["leg2"]["phi"]["entire"].values()
    hpt_pass_barrel_both_leg1, hpt_all_barrel_both_leg1 = histograms_both["leg1"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_both_leg1, hpt_all_endcap_both_leg1 = histograms_both["leg1"]["pt"][
        "endcap"
    ].values()
    heta_pass_both_leg1, heta_all_both_leg1 = histograms_both["leg1"]["eta"][
        "entire"
    ].values()
    hphi_pass_both_leg1, hphi_all_both_leg1 = histograms_both["leg1"]["phi"][
        "entire"
    ].values()
    hpt_pass_barrel_both_leg2, hpt_all_barrel_both_leg2 = histograms_both["leg2"]["pt"][
        "barrel"
    ].values()
    hpt_pass_endcap_both_leg2, hpt_all_endcap_both_leg2 = histograms_both["leg2"]["pt"][
        "endcap"
    ].values()
    heta_pass_both_leg2, heta_all_both_leg2 = histograms_both["leg2"]["eta"][
        "entire"
    ].values()
    hphi_pass_both_leg2, hphi_all_both_leg2 = histograms_both["leg2"]["phi"][
        "entire"
    ].values()

    assert np.all(
        hpt_pass_barrel_leg1.values(flow=True)
        == hpt_pass_barrel_both_leg1.values(flow=True)
    )
    assert np.all(
        hpt_pass_endcap_leg1.values(flow=True)
        == hpt_pass_endcap_both_leg1.values(flow=True)
    )
    assert np.all(
        heta_pass_leg1.values(flow=True) == heta_pass_both_leg1.values(flow=True)
    )
    assert np.all(
        hphi_pass_leg1.values(flow=True) == hphi_pass_both_leg1.values(flow=True)
    )
    assert np.all(
        hpt_pass_barrel_leg2.values(flow=True)
        == hpt_pass_barrel_both_leg2.values(flow=True)
    )
    assert np.all(
        hpt_pass_endcap_leg2.values(flow=True)
        == hpt_pass_endcap_both_leg2.values(flow=True)
    )
    assert np.all(
        heta_pass_leg2.values(flow=True) == heta_pass_both_leg2.values(flow=True)
    )
    assert np.all(
        hphi_pass_leg2.values(flow=True) == hphi_pass_both_leg2.values(flow=True)
    )

    assert hpt_pass_barrel_leg1.sum(flow=True) == hpt_pass_barrel_both_leg1.sum(
        flow=True
    )
    assert hpt_pass_endcap_leg1.sum(flow=True) == hpt_pass_endcap_both_leg1.sum(
        flow=True
    )
    assert heta_pass_leg1.sum(flow=True) == heta_pass_both_leg1.sum(flow=True)
    assert hphi_pass_leg1.sum(flow=True) == hphi_pass_both_leg1.sum(flow=True)
    assert hpt_pass_barrel_leg2.sum(flow=True) == hpt_pass_barrel_both_leg2.sum(
        flow=True
    )
    assert hpt_pass_endcap_leg2.sum(flow=True) == hpt_pass_endcap_both_leg2.sum(
        flow=True
    )
    assert heta_pass_leg2.sum(flow=True) == heta_pass_both_leg2.sum(flow=True)
    assert hphi_pass_leg2.sum(flow=True) == hphi_pass_both_leg2.sum(flow=True)

    assert (
        hpt_pass_barrel_leg1.sum(flow=True) + hpt_pass_endcap_leg1.sum(flow=True) == 0.0
    )
    assert (
        hpt_all_barrel_leg1.sum(flow=True) + hpt_all_endcap_leg1.sum(flow=True) == 893.0
    )
    assert heta_pass_leg1.sum(flow=True) == 0.0
    assert heta_all_leg1.sum(flow=True) == 893.0
    assert hphi_pass_leg1.sum(flow=True) == 0.0
    assert hphi_all_leg1.sum(flow=True) == 893.0
    assert (
        hpt_pass_barrel_leg2.sum(flow=True) + hpt_pass_endcap_leg2.sum(flow=True) == 0.0
    )
    assert (
        hpt_all_barrel_leg2.sum(flow=True) + hpt_all_endcap_leg2.sum(flow=True) == 893.0
    )
    assert heta_pass_leg2.sum(flow=True) == 0.0
    assert heta_all_leg2.sum(flow=True) == 893.0
    assert hphi_pass_leg2.sum(flow=True) == 0.0
    assert hphi_all_leg2.sum(flow=True) == 893.0

    assert (
        hpt_pass_barrel_leg1.values(flow=True)[0]
        + hpt_pass_endcap_leg1.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel_leg1.values(flow=True)[0]
        + hpt_all_endcap_leg1.values(flow=True)[0]
        == 0.0
    )
    assert heta_pass_leg1.values(flow=True)[0] == 0.0
    assert heta_all_leg1.values(flow=True)[0] == 0.0
    assert hphi_pass_leg1.values(flow=True)[0] == 0.0
    assert hphi_all_leg1.values(flow=True)[0] == 0.0
    assert (
        hpt_pass_barrel_leg2.values(flow=True)[0]
        + hpt_pass_endcap_leg2.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel_leg2.values(flow=True)[0]
        + hpt_all_endcap_leg2.values(flow=True)[0]
        == 0.0
    )
    assert heta_pass_leg2.values(flow=True)[0] == 0.0
    assert heta_all_leg2.values(flow=True)[0] == 0.0
    assert hphi_pass_leg2.values(flow=True)[0] == 0.0
    assert hphi_all_leg2.values(flow=True)[0] == 0.0


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
        fileset = {
            "sample": {
                "files": {os.path.abspath("tests/samples/DYto2E.root"): "Events"}
            }
        }

    if do_preprocess:
        if allow_read_errors_with_report:
            with pytest.raises(FileNotFoundError):
                preprocess(fileset)
            fileset_available, fileset_updated = preprocess(
                fileset, skip_bad_files=True
            )
            fileset = fileset_available

    tag_n_probe = DoubleElePt_CaloIdL_MW(
        fileset,
        33,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
    )

    with Client():
        res_leg1 = tag_n_probe.get_tnp_arrays(
            leg="first",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        res_leg2 = tag_n_probe.get_tnp_arrays(
            leg="second",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        res_both = tag_n_probe.get_tnp_arrays(
            leg="both",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        if allow_read_errors_with_report:
            arrays_leg1 = res_leg1[0]["sample"]
            report_arrays_leg1 = res_leg1[1]["sample"]
            arrays_leg2 = res_leg2[0]["sample"]
            report_arrays_leg2 = res_leg2[1]["sample"]
            arrays_both = res_both[0]["sample"]
            report_arrays_both = res_both[1]["sample"]
            if not do_preprocess:
                assert report_arrays_leg1.exception[1] == "FileNotFoundError"
                assert report_arrays_leg2.exception[1] == "FileNotFoundError"
                assert report_arrays_both.exception[1] == "FileNotFoundError"
        else:
            arrays_leg1 = res_leg1["sample"]
            arrays_leg2 = res_leg2["sample"]
            arrays_both = res_both["sample"]

        res_leg1 = tag_n_probe.get_tnp_histograms(
            leg="first",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        res_leg2 = tag_n_probe.get_tnp_histograms(
            leg="second",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        res_both = tag_n_probe.get_tnp_histograms(
            leg="both",
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=True,
            scheduler=None,
            progress=False,
        )
        if allow_read_errors_with_report:
            histograms_leg1 = res_leg1[0]["sample"]
            histograms_leg2 = res_leg2[0]["sample"]
            histograms_both = res_both[0]["sample"]
        else:
            histograms_leg1 = res_leg1["sample"]
            histograms_leg2 = res_leg2["sample"]
            histograms_both = res_both["sample"]
        if allow_read_errors_with_report:
            histograms_leg1 = res_leg1[0]["sample"]
            report_histograms_leg1 = res_leg1[1]["sample"]
            histograms_leg2 = res_leg2[0]["sample"]
            report_histograms_leg2 = res_leg2[1]["sample"]
            histograms_both = res_both[0]["sample"]
            report_histograms_both = res_both[1]["sample"]
            if not do_preprocess:
                assert report_histograms_leg1.exception[1] == "FileNotFoundError"
                assert report_histograms_leg2.exception[1] == "FileNotFoundError"
                assert report_histograms_both.exception[1] == "FileNotFoundError"
        else:
            histograms_leg1 = res_leg1["sample"]
            histograms_leg2 = res_leg2["sample"]
            histograms_both = res_both["sample"]

        arrays_pass_leg1, arrays_all_leg1 = arrays_leg1["leg1"]
        arrays_pass_leg2, arrays_all_leg2 = arrays_leg2["leg2"]
        arrays_pass_both_leg1, arrays_all_both_leg1 = arrays_both["leg1"]
        arrays_pass_both_leg2, arrays_all_both_leg2 = arrays_both["leg2"]

        for field in ["pt", "eta", "phi"]:
            assert np.all(arrays_pass_leg1[field] == arrays_pass_both_leg1[field])
            assert np.all(arrays_pass_leg2[field] == arrays_pass_both_leg2[field])
            assert np.all(arrays_all_leg1[field] == arrays_all_both_leg1[field])
            assert np.all(arrays_all_leg2[field] == arrays_all_both_leg2[field])

        if not preprocess:
            assert report_arrays_leg1.exception[1] == "FileNotFoundError"
            assert report_arrays_leg2.exception[1] == "FileNotFoundError"
            assert report_arrays_both.exception[1] == "FileNotFoundError"
            assert report_histograms_leg1.exception[1] == "FileNotFoundError"
            assert report_histograms_leg2.exception[1] == "FileNotFoundError"
            assert report_histograms_both.exception[1] == "FileNotFoundError"

        hpt_pass_barrel_leg1, hpt_all_barrel_leg1 = histograms_leg1["leg1"]["pt"][
            "barrel"
        ].values()
        hpt_pass_endcap_leg1, hpt_all_endcap_leg1 = histograms_leg1["leg1"]["pt"][
            "endcap"
        ].values()
        heta_pass_leg1, heta_all_leg1 = histograms_leg1["leg1"]["eta"][
            "entire"
        ].values()
        hphi_pass_leg1, hphi_all_leg1 = histograms_leg1["leg1"]["phi"][
            "entire"
        ].values()
        hpt_pass_barrel_leg2, hpt_all_barrel_leg2 = histograms_leg2["leg2"]["pt"][
            "barrel"
        ].values()
        hpt_pass_endcap_leg2, hpt_all_endcap_leg2 = histograms_leg2["leg2"]["pt"][
            "endcap"
        ].values()
        heta_pass_leg2, heta_all_leg2 = histograms_leg2["leg2"]["eta"][
            "entire"
        ].values()
        hphi_pass_leg2, hphi_all_leg2 = histograms_leg2["leg2"]["phi"][
            "entire"
        ].values()
        hpt_pass_barrel_both_leg1, hpt_all_barrel_both_leg1 = histograms_both["leg1"][
            "pt"
        ]["barrel"].values()
        hpt_pass_endcap_both_leg1, hpt_all_endcap_both_leg1 = histograms_both["leg1"][
            "pt"
        ]["endcap"].values()
        heta_pass_both_leg1, heta_all_both_leg1 = histograms_both["leg1"]["eta"][
            "entire"
        ].values()
        hphi_pass_both_leg1, hphi_all_both_leg1 = histograms_both["leg1"]["phi"][
            "entire"
        ].values()
        hpt_pass_barrel_both_leg2, hpt_all_barrel_both_leg2 = histograms_both["leg2"][
            "pt"
        ]["barrel"].values()
        hpt_pass_endcap_both_leg2, hpt_all_endcap_both_leg2 = histograms_both["leg2"][
            "pt"
        ]["endcap"].values()
        heta_pass_both_leg2, heta_all_both_leg2 = histograms_both["leg2"]["eta"][
            "entire"
        ].values()
        hphi_pass_both_leg2, hphi_all_both_leg2 = histograms_both["leg2"]["phi"][
            "entire"
        ].values()

        assert np.all(
            hpt_pass_barrel_leg1.values(flow=True)
            == hpt_pass_barrel_both_leg1.values(flow=True)
        )
        assert np.all(
            hpt_pass_endcap_leg1.values(flow=True)
            == hpt_pass_endcap_both_leg1.values(flow=True)
        )
        assert np.all(
            heta_pass_leg1.values(flow=True) == heta_pass_both_leg1.values(flow=True)
        )
        assert np.all(
            hphi_pass_leg1.values(flow=True) == hphi_pass_both_leg1.values(flow=True)
        )
        assert np.all(
            hpt_pass_barrel_leg2.values(flow=True)
            == hpt_pass_barrel_both_leg2.values(flow=True)
        )
        assert np.all(
            hpt_pass_endcap_leg2.values(flow=True)
            == hpt_pass_endcap_both_leg2.values(flow=True)
        )
        assert np.all(
            heta_pass_leg2.values(flow=True) == heta_pass_both_leg2.values(flow=True)
        )
        assert np.all(
            hphi_pass_leg2.values(flow=True) == hphi_pass_both_leg2.values(flow=True)
        )

        assert hpt_pass_barrel_leg1.sum(flow=True) == hpt_pass_barrel_both_leg1.sum(
            flow=True
        )
        assert hpt_pass_endcap_leg1.sum(flow=True) == hpt_pass_endcap_both_leg1.sum(
            flow=True
        )
        assert heta_pass_leg1.sum(flow=True) == heta_pass_both_leg1.sum(flow=True)
        assert hphi_pass_leg1.sum(flow=True) == hphi_pass_both_leg1.sum(flow=True)
        assert hpt_pass_barrel_leg2.sum(flow=True) == hpt_pass_barrel_both_leg2.sum(
            flow=True
        )
        assert hpt_pass_endcap_leg2.sum(flow=True) == hpt_pass_endcap_both_leg2.sum(
            flow=True
        )
        assert heta_pass_leg2.sum(flow=True) == heta_pass_both_leg2.sum(flow=True)
        assert hphi_pass_leg2.sum(flow=True) == hphi_pass_both_leg2.sum(flow=True)

        assert (
            hpt_pass_barrel_leg1.sum(flow=True) + hpt_pass_endcap_leg1.sum(flow=True)
            == 0.0
        )
        assert (
            hpt_all_barrel_leg1.sum(flow=True) + hpt_all_endcap_leg1.sum(flow=True)
            == 893.0
        )
        assert heta_pass_leg1.sum(flow=True) == 0.0
        assert heta_all_leg1.sum(flow=True) == 893.0
        assert hphi_pass_leg1.sum(flow=True) == 0.0
        assert hphi_all_leg1.sum(flow=True) == 893.0
        assert (
            hpt_pass_barrel_leg2.sum(flow=True) + hpt_pass_endcap_leg2.sum(flow=True)
            == 0.0
        )
        assert (
            hpt_all_barrel_leg2.sum(flow=True) + hpt_all_endcap_leg2.sum(flow=True)
            == 893.0
        )
        assert heta_pass_leg2.sum(flow=True) == 0.0
        assert heta_all_leg2.sum(flow=True) == 893.0
        assert hphi_pass_leg2.sum(flow=True) == 0.0
        assert hphi_all_leg2.sum(flow=True) == 893.0

        assert (
            hpt_pass_barrel_leg1.values(flow=True)[0]
            + hpt_pass_endcap_leg1.values(flow=True)[0]
            == 0.0
        )
        assert (
            hpt_all_barrel_leg1.values(flow=True)[0]
            + hpt_all_endcap_leg1.values(flow=True)[0]
            == 0.0
        )
        assert heta_pass_leg1.values(flow=True)[0] == 0.0
        assert heta_all_leg1.values(flow=True)[0] == 0.0
        assert hphi_pass_leg1.values(flow=True)[0] == 0.0
        assert hphi_all_leg1.values(flow=True)[0] == 0.0
        assert (
            hpt_pass_barrel_leg2.values(flow=True)[0]
            + hpt_pass_endcap_leg2.values(flow=True)[0]
            == 0.0
        )
        assert (
            hpt_all_barrel_leg2.values(flow=True)[0]
            + hpt_all_endcap_leg2.values(flow=True)[0]
            == 0.0
        )
        assert heta_pass_leg2.values(flow=True)[0] == 0.0
        assert heta_all_leg2.values(flow=True)[0] == 0.0
        assert hphi_pass_leg2.values(flow=True)[0] == 0.0
        assert hphi_all_leg2.values(flow=True)[0] == 0.0
