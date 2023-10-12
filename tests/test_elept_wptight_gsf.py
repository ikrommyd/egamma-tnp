import os

import pytest

from egamma_tnp.triggers import ElePt_WPTight_Gsf


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_local_compute(scheduler, preprocess):
    tag_n_probe = ElePt_WPTight_Gsf(
        os.path.abspath("tests/samples/DYto2E.root"),
        32,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events()

    histograms = tag_n_probe.get_tnp_histograms(
        eta_regions={"barrel": [0.0, 1.4442], "endcap": [1.566, 2.5]},
        compute=True,
        scheduler=scheduler,
        progress=True,
    )
    (
        hpt_pass_barrel,
        hpt_all_barrel,
        heta_pass_barrel,
        heta_all_barrel,
        hphi_pass_barrel,
        hphi_all_barrel,
    ) = histograms["barrel"]
    (
        hpt_pass_endcap,
        hpt_all_endcap,
        heta_pass_endcap,
        heta_all_endcap,
        hphi_pass_endcap,
        hphi_all_endcap,
    ) = histograms["endcap"]

    assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 144.0
    assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 171.0
    assert heta_pass_barrel.sum(flow=True) + heta_pass_endcap.sum(flow=True) == 144.0
    assert heta_all_barrel.sum(flow=True) + heta_all_endcap.sum(flow=True) == 171.0
    assert hphi_pass_barrel.sum(flow=True) + hphi_pass_endcap.sum(flow=True) == 144.0
    assert hphi_all_barrel.sum(flow=True) + hphi_all_endcap.sum(flow=True) == 171.0

    assert (
        hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0]
        == 0.0
    )
    assert (
        hpt_all_barrel.values(flow=True)[0] + hpt_all_endcap.values(flow=True)[0] == 0.0
    )
    assert (
        heta_pass_barrel.values(flow=True)[0] + heta_pass_endcap.values(flow=True)[0]
        == 0.0
    )
    assert (
        heta_all_barrel.values(flow=True)[0] + heta_all_endcap.values(flow=True)[0]
        == 0.0
    )
    assert (
        hphi_pass_barrel.values(flow=True)[0] + hphi_pass_endcap.values(flow=True)[0]
        == 0.0
    )
    assert (
        hphi_all_barrel.values(flow=True)[0] + hphi_all_endcap.values(flow=True)[0]
        == 0.0
    )


@pytest.mark.parametrize("preprocess", [False, True])
def test_distributed_compute(preprocess):
    from distributed import Client

    tag_n_probe = ElePt_WPTight_Gsf(
        os.path.abspath("tests/samples/DYto2E.root"),
        32,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events()

    with Client():
        histograms = tag_n_probe.get_tnp_histograms(
            eta_regions={"barrel": [0.0, 1.4442], "endcap": [1.566, 2.5]},
            compute=True,
            scheduler=None,
            progress=True,
        )
        (
            hpt_pass_barrel,
            hpt_all_barrel,
            heta_pass_barrel,
            heta_all_barrel,
            hphi_pass_barrel,
            hphi_all_barrel,
        ) = histograms["barrel"]
        (
            hpt_pass_endcap,
            hpt_all_endcap,
            heta_pass_endcap,
            heta_all_endcap,
            hphi_pass_endcap,
            hphi_all_endcap,
        ) = histograms["endcap"]

        assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 144.0
        assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 171.0
        assert (
            heta_pass_barrel.sum(flow=True) + heta_pass_endcap.sum(flow=True) == 144.0
        )
        assert heta_all_barrel.sum(flow=True) + heta_all_endcap.sum(flow=True) == 171.0
        assert (
            hphi_pass_barrel.sum(flow=True) + hphi_pass_endcap.sum(flow=True) == 144.0
        )
        assert hphi_all_barrel.sum(flow=True) + hphi_all_endcap.sum(flow=True) == 171.0

        assert (
            hpt_pass_barrel.values(flow=True)[0] + hpt_pass_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            hpt_all_barrel.values(flow=True)[0] + hpt_all_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            heta_pass_barrel.values(flow=True)[0]
            + heta_pass_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            heta_all_barrel.values(flow=True)[0] + heta_all_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            hphi_pass_barrel.values(flow=True)[0]
            + hphi_pass_endcap.values(flow=True)[0]
            == 0.0
        )
        assert (
            hphi_all_barrel.values(flow=True)[0] + hphi_all_endcap.values(flow=True)[0]
            == 0.0
        )
