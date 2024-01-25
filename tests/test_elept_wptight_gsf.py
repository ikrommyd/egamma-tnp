import os

import pytest
from coffea.nanoevents import NanoAODSchema

from egamma_tnp.triggers import ElePt_WPTight_Gsf

NanoAODSchema.error_missing_event_ids = False


fileset = {
    "sample": {
        "files": {
            os.path.abspath("tests/samples/DYto2E.root"): "Events",
            os.path.abspath("tests/samples/not_a_file.root"): "Events",
        }
    }
}


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
def test_local_compute(scheduler):
    tag_n_probe = ElePt_WPTight_Gsf(
        fileset,
        32,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
    )

    res = tag_n_probe.get_tnp_histograms(
        uproot_options={"allow_read_errors_with_report": True},
        compute=True,
        scheduler=scheduler,
        progress=True,
    )
    histograms = res[0]["sample"]
    report = res[1]["sample"]

    assert report.exception[1] == "FileNotFoundError"

    hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
    hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
    heta_pass, heta_all = histograms["eta"]["entire"].values()
    hphi_pass, hphi_all = histograms["phi"]["entire"].values()

    assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 141.0
    assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 167.0
    assert heta_pass.sum(flow=True) == 141.0
    assert heta_all.sum(flow=True) == 167.0
    assert hphi_pass.sum(flow=True) == 141.0
    assert hphi_all.sum(flow=True) == 167.0

    assert (
        hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel.values(flow=True)[0] + hpt_all_endcap.values(flow=True)[0] == 0.0
    )
    assert heta_pass.values(flow=True)[0] == 0.0
    assert heta_all.values(flow=True)[0] == 0.0
    assert hphi_pass.values(flow=True)[0] == 0.0
    assert hphi_all.values(flow=True)[0] == 0.0


def test_distributed_compute():
    from distributed import Client

    tag_n_probe = ElePt_WPTight_Gsf(
        fileset,
        32,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
    )

    with Client():
        res = tag_n_probe.get_tnp_histograms(
            uproot_options={"allow_read_errors_with_report": True},
            compute=True,
            scheduler=None,
            progress=True,
        )
        histograms = res[0]["sample"]
        report = res[1]["sample"]

        assert report.exception[1] == "FileNotFoundError"

        hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
        hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
        heta_pass, heta_all = histograms["eta"]["entire"].values()
        hphi_pass, hphi_all = histograms["phi"]["entire"].values()

        assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 141.0
        assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 167.0
        assert heta_pass.sum(flow=True) == 141.0
        assert heta_all.sum(flow=True) == 167.0
        assert hphi_pass.sum(flow=True) == 141.0
        assert hphi_all.sum(flow=True) == 167.0

        assert (
            hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            hpt_all_barrel.values(flow=True)[0] + hpt_all_endcap.values(flow=True)[0]
            == 0.0
        )
        assert heta_pass.values(flow=True)[0] == 0.0
        assert heta_all.values(flow=True)[0] == 0.0
        assert hphi_pass.values(flow=True)[0] == 0.0
        assert hphi_all.values(flow=True)[0] == 0.0
