import os

import pytest

from egamma_tnp.triggers import ElePt_CaloIdVT_GsfTrkIdT


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_local_compute(scheduler, preprocess):
    tag_n_probe = ElePt_CaloIdVT_GsfTrkIdT(
        os.path.abspath("tests/samples/DYto2E.root"),
        115,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events()

    (
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
        hphi_pass,
        hphi_all,
    ) = tag_n_probe.get_tnp_histograms(
        compute=True, scheduler=scheduler, progress=True
    )[
        "all"
    ]

    assert hpt_pass.sum(flow=True) == 0.0
    assert hpt_all.sum(flow=True) == 0.0
    assert heta_pass.sum(flow=True) == 0.0
    assert heta_all.sum(flow=True) == 0.0
    assert hphi_pass.sum(flow=True) == 0.0
    assert hphi_all.sum(flow=True) == 0.0

    assert hpt_pass.values(flow=True)[0] == 0.0
    assert hpt_all.values(flow=True)[0] == 0.0
    assert heta_pass.values(flow=True)[0] == 0.0
    assert heta_all.values(flow=True)[0] == 0.0
    assert hphi_pass.values(flow=True)[0] == 0.0
    assert hphi_all.values(flow=True)[0] == 0.0


@pytest.mark.parametrize("preprocess", [False, True])
def test_distributed_compute(preprocess):
    from distributed import Client

    tag_n_probe = ElePt_CaloIdVT_GsfTrkIdT(
        os.path.abspath("tests/samples/DYto2E.root"),
        115,
        goldenjson=None,
        toquery=False,
        redirect=False,
        preprocess=preprocess,
    )
    tag_n_probe.load_events()

    with Client():
        (
            hpt_pass,
            hpt_all,
            heta_pass,
            heta_all,
            hphi_pass,
            hphi_all,
        ) = tag_n_probe.get_tnp_histograms(compute=True, scheduler=None, progress=True)[
            "all"
        ]

        assert hpt_pass.sum(flow=True) == 0.0
        assert hpt_all.sum(flow=True) == 0.0
        assert heta_pass.sum(flow=True) == 0.0
        assert heta_all.sum(flow=True) == 0.0
        assert hphi_pass.sum(flow=True) == 0.0
        assert hphi_all.sum(flow=True) == 0.0

        assert hpt_pass.values(flow=True)[0] == 0.0
        assert hpt_all.values(flow=True)[0] == 0.0
        assert heta_pass.values(flow=True)[0] == 0.0
        assert heta_all.values(flow=True)[0] == 0.0
        assert hphi_pass.values(flow=True)[0] == 0.0
        assert hphi_all.values(flow=True)[0] == 0.0
