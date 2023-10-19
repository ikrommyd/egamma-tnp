import os

import pytest
from coffea.nanoevents import NanoAODSchema

from egamma_tnp.triggers import ElePt_WPTight_Gsf

NanoAODSchema.error_missing_event_ids = False


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_local_compute(scheduler, preprocess):
    tag_n_probe = ElePt_WPTight_Gsf(
        os.path.abspath("tests/samples/DYto2E.root"),
        32,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events(from_root_args={"schemaclass": NanoAODSchema})

    histograms = tag_n_probe.get_tnp_histograms(
        compute=True,
        scheduler=scheduler,
        progress=True,
    )

    hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
    hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
    heta_pass, heta_all = histograms["eta"]["entire"].values()
    hphi_pass, hphi_all = histograms["phi"]["entire"].values()

    assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 144.0
    assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 171.0
    assert heta_pass.sum(flow=True) == 144.0
    assert heta_all.sum(flow=True) == 171.0
    assert hphi_pass.sum(flow=True) == 144.0

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


@pytest.mark.parametrize("preprocess", [False, True])
def test_distributed_compute(preprocess):
    from distributed import Client

    tag_n_probe = ElePt_WPTight_Gsf(
        os.path.abspath("tests/samples/DYto2E.root"),
        32,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events(from_root_args={"schemaclass": NanoAODSchema})

    with Client():
        histograms = tag_n_probe.get_tnp_histograms(
            compute=True,
            scheduler=None,
            progress=True,
        )

        hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
        hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
        heta_pass, heta_all = histograms["eta"]["entire"].values()
        hphi_pass, hphi_all = histograms["phi"]["entire"].values()

        assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 144.0
        assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 171.0
        assert heta_pass.sum(flow=True) == 144.0
        assert heta_all.sum(flow=True) == 171.0
        assert hphi_pass.sum(flow=True) == 144.0

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
