import os

import pytest
from coffea.dataset_tools import preprocess

from egamma_tnp.triggers import TagNProbeFromNTuples


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_without_compute(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree",
                    os.path.abspath("tests/samples/not_a_file.root"): "fitter_tree",
                }
            }
        }
    else:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree"
                }
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

    tag_n_probe_leg1 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWSeedLegL1match",
        trigger_pt=5,
        goldenjson=None,
    )
    tag_n_probe_leg2 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWUnsLeg",
        trigger_pt=5,
        goldenjson=None,
    )

    for tag_n_probe in [tag_n_probe_leg1, tag_n_probe_leg2]:
        res = tag_n_probe.get_tnp_histograms(
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
            compute=False,
            scheduler=None,
            progress=False,
        )

        if allow_read_errors_with_report:
            histograms = res[0]["sample"]
        else:
            histograms = res["sample"]

        hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
        hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
        heta_pass, heta_all = histograms["eta"]["entire"].values()
        hphi_pass, hphi_all = histograms["phi"]["entire"].values()

        assert hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == 0.0
        assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 0.0
        assert heta_pass.sum(flow=True) == 0.0
        assert heta_all.sum(flow=True) == 0.0
        assert hphi_pass.sum(flow=True) == 0.0
        assert hphi_all.sum(flow=True) == 0.0

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


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_local_compute(do_preprocess, allow_read_errors_with_report):
    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree",
                    os.path.abspath("tests/samples/not_a_file.root"): "fitter_tree",
                }
            }
        }
    else:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree"
                }
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

    tag_n_probe_leg1 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWSeedLegL1match",
        trigger_pt=5,
        goldenjson=None,
    )
    tag_n_probe_leg2 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWUnsLeg",
        trigger_pt=5,
        goldenjson=None,
    )

    for tag_n_probe, target_pt, target_eta_phi in zip(
        [tag_n_probe_leg1, tag_n_probe_leg2], [361.0, 367.0], [372.0, 378.0]
    ):
        res = tag_n_probe.get_tnp_histograms(
            uproot_options={
                "allow_read_errors_with_report": allow_read_errors_with_report
            },
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

        hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
        hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
        heta_pass, heta_all = histograms["eta"]["entire"].values()
        hphi_pass, hphi_all = histograms["phi"]["entire"].values()

        assert (
            hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True) == target_pt
        )
        assert hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 490.0
        assert heta_pass.sum(flow=True) == target_eta_phi
        assert heta_all.sum(flow=True) == 505.0
        assert hphi_pass.sum(flow=True) == target_eta_phi
        assert hphi_all.sum(flow=True) == 505.0

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


@pytest.mark.parametrize("do_preprocess", [True, False])
@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_distributed_compute(do_preprocess, allow_read_errors_with_report):
    from distributed import Client

    if allow_read_errors_with_report:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree",
                    os.path.abspath("tests/samples/not_a_file.root"): "fitter_tree",
                }
            }
        }
    else:
        fileset = {
            "sample": {
                "files": {
                    os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree"
                }
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

    tag_n_probe_leg1 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWSeedLegL1match",
        trigger_pt=5,
        goldenjson=None,
    )
    tag_n_probe_leg2 = TagNProbeFromNTuples(
        fileset,
        "passHltDoubleEle33CaloIdLMWUnsLeg",
        trigger_pt=5,
        goldenjson=None,
    )

    with Client():
        for tag_n_probe, target_pt, target_eta_phi in zip(
            [tag_n_probe_leg1, tag_n_probe_leg2], [361.0, 367.0], [372.0, 378.0]
        ):
            res = tag_n_probe.get_tnp_histograms(
                uproot_options={
                    "allow_read_errors_with_report": allow_read_errors_with_report
                },
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

            hpt_pass_barrel, hpt_all_barrel = histograms["pt"]["barrel"].values()
            hpt_pass_endcap, hpt_all_endcap = histograms["pt"]["endcap"].values()
            heta_pass, heta_all = histograms["eta"]["entire"].values()
            hphi_pass, hphi_all = histograms["phi"]["entire"].values()

            assert (
                hpt_pass_barrel.sum(flow=True) + hpt_pass_endcap.sum(flow=True)
                == target_pt
            )
            assert (
                hpt_all_barrel.sum(flow=True) + hpt_all_endcap.sum(flow=True) == 490.0
            )
            assert heta_pass.sum(flow=True) == target_eta_phi
            assert heta_all.sum(flow=True) == 505.0
            assert hphi_pass.sum(flow=True) == target_eta_phi
            assert hphi_all.sum(flow=True) == 505.0

            assert (
                hpt_pass_barrel.values(flow=True)[0]
                + hpt_pass_endcap.values(flow=True)[0]
                == 0.0
            )
            assert (
                hpt_all_barrel.values(flow=True)[0]
                + hpt_all_endcap.values(flow=True)[0]
                == 0.0
            )
            assert heta_pass.values(flow=True)[0] == 0.0
            assert heta_all.values(flow=True)[0] == 0.0
            assert hphi_pass.values(flow=True)[0] == 0.0
            assert hphi_all.values(flow=True)[0] == 0.0
