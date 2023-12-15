import os

import numpy as np
import pytest
from coffea.nanoevents import NanoAODSchema

from egamma_tnp.triggers import ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL

NanoAODSchema.error_missing_event_ids = False


@pytest.mark.parametrize("scheduler", ["threads", "processes", "single-threaded"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_local_compute(scheduler, preprocess):
    files = [os.path.abspath("tests/samples/DYto2E.root")]
    if not preprocess:
        files.append(os.path.abspath("tests/samples/not_a_file.root"))
    tag_n_probe = ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL(
        files,
        23,
        12,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
        toquery=False,
        redirector=None,
        preprocess=preprocess,
    )
    tag_n_probe.load_events(
        from_root_args={"schemaclass": NanoAODSchema},
        allow_read_errors_with_report=True,
    )

    arrays_leg1, report_arrays_leg1 = tag_n_probe.get_arrays(
        leg="first",
        compute=True,
        progress=True,
    )
    arrays_leg2, report_arrays_leg2 = tag_n_probe.get_arrays(
        leg="second",
        compute=True,
        progress=True,
    )
    arrays_both, report_arrays_both = tag_n_probe.get_arrays(
        leg="both",
        compute=True,
        progress=True,
    )
    histograms_leg1, report_histograms_leg1 = tag_n_probe.get_tnp_histograms(
        leg="first",
        compute=True,
        progress=True,
    )
    histograms_leg2, report_histograms_leg2 = tag_n_probe.get_tnp_histograms(
        leg="second",
        compute=True,
        progress=True,
    )
    histograms_both, report_histograms_both = tag_n_probe.get_tnp_histograms(
        leg="both",
        compute=True,
        progress=True,
    )

    for arr1, arr2 in zip(arrays_leg1["leg1"], arrays_both["leg1"]):
        assert np.all(arr1 == arr2)
    for arr1, arr2 in zip(arrays_leg2["leg2"], arrays_both["leg2"]):
        assert np.all(arr1 == arr2)

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
        hpt_pass_barrel_leg1.sum(flow=True) + hpt_pass_endcap_leg1.sum(flow=True)
        == 181.0
    )
    assert (
        hpt_all_barrel_leg1.sum(flow=True) + hpt_all_endcap_leg1.sum(flow=True) == 190.0
    )
    assert heta_pass_leg1.sum(flow=True) == 181.0
    assert heta_all_leg1.sum(flow=True) == 190.0
    assert hphi_pass_leg1.sum(flow=True) == 181.0
    assert hphi_all_leg1.sum(flow=True) == 190.0
    assert (
        hpt_pass_barrel_leg2.sum(flow=True) + hpt_pass_endcap_leg2.sum(flow=True)
        == 188.0
    )
    assert (
        hpt_all_barrel_leg2.sum(flow=True) + hpt_all_endcap_leg2.sum(flow=True) == 197.0
    )
    assert heta_pass_leg2.sum(flow=True) == 188.0
    assert heta_all_leg2.sum(flow=True) == 197.0
    assert hphi_pass_leg2.sum(flow=True) == 188.0
    assert hphi_all_leg2.sum(flow=True) == 197.0

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


@pytest.mark.parametrize("preprocess", [False, True])
def test_distributed_compute(preprocess):
    from distributed import Client

    files = [os.path.abspath("tests/samples/DYto2E.root")]
    if not preprocess:
        files.append(os.path.abspath("tests/samples/not_a_file.root"))
    tag_n_probe = ElePt1_ElePt2_CaloIdL_TrackIdL_IsoVL(
        files,
        23,
        12,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=True,
        goldenjson=None,
        toquery=False,
        redirector=None,
        preprocess=preprocess,
    )
    tag_n_probe.load_events(
        from_root_args={"schemaclass": NanoAODSchema},
        allow_read_errors_with_report=True,
    )

    with Client():
        arrays_leg1, report_arrays_leg1 = tag_n_probe.get_arrays(
            leg="first",
            compute=True,
            progress=True,
        )
        arrays_leg2, report_arrays_leg2 = tag_n_probe.get_arrays(
            leg="second",
            compute=True,
            progress=True,
        )
        arrays_both, report_arrays_both = tag_n_probe.get_arrays(
            leg="both",
            compute=True,
            progress=True,
        )
        histograms_leg1, report_histograms_leg1 = tag_n_probe.get_tnp_histograms(
            leg="first",
            compute=True,
            progress=True,
        )
        histograms_leg2, report_histograms_leg2 = tag_n_probe.get_tnp_histograms(
            leg="second",
            compute=True,
            progress=True,
        )
        histograms_both, report_histograms_both = tag_n_probe.get_tnp_histograms(
            leg="both",
            compute=True,
            progress=True,
        )

        for arr1, arr2 in zip(arrays_leg1["leg1"], arrays_both["leg1"]):
            assert np.all(arr1 == arr2)
        for arr1, arr2 in zip(arrays_leg2["leg2"], arrays_both["leg2"]):
            assert np.all(arr1 == arr2)

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
            == 181.0
        )
        assert (
            hpt_all_barrel_leg1.sum(flow=True) + hpt_all_endcap_leg1.sum(flow=True)
            == 190.0
        )
        assert heta_pass_leg1.sum(flow=True) == 181.0
        assert heta_all_leg1.sum(flow=True) == 190.0
        assert hphi_pass_leg1.sum(flow=True) == 181.0
        assert hphi_all_leg1.sum(flow=True) == 190.0
        assert (
            hpt_pass_barrel_leg2.sum(flow=True) + hpt_pass_endcap_leg2.sum(flow=True)
            == 188.0
        )
        assert (
            hpt_all_barrel_leg2.sum(flow=True) + hpt_all_endcap_leg2.sum(flow=True)
            == 197.0
        )
        assert heta_pass_leg2.sum(flow=True) == 188.0
        assert heta_all_leg2.sum(flow=True) == 197.0
        assert hphi_pass_leg2.sum(flow=True) == 188.0
        assert hphi_all_leg2.sum(flow=True) == 197.0

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
