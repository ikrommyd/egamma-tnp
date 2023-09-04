import os

import pytest

from egamma_tnp import TagNProbe


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_local_compute(scheduler, preprocess):
    tag_n_probe = TagNProbe(
        [
            os.path.abspath("root_files/Egamma0.root"),
            os.path.abspath("root_files/Egamma1.root"),
        ],
        32,
        goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
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
    ) = tag_n_probe.get_tnp_histograms(compute=True, scheduler=scheduler, progress=True)

    assert hpt_pass.sum(flow=True) == 14598.0
    assert hpt_all.sum(flow=True) == 16896.0


@pytest.mark.parametrize("preprocess", [False, True])
def test_distributed_compute(preprocess):
    from distributed import Client

    tag_n_probe = TagNProbe(
        [
            os.path.abspath("root_files/Egamma0.root"),
            os.path.abspath("root_files/Egamma1.root"),
        ],
        32,
        goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
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
        ) = tag_n_probe.get_tnp_histograms(compute=True, scheduler=None, progress=True)

        assert hpt_pass.sum(flow=True) == 14598.0
        assert hpt_all.sum(flow=True) == 16896.0
