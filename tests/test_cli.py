from __future__ import annotations

import json
import os
import pickle
import subprocess

import awkward as ak
import dask
import numpy as np
from dask.diagnostics import ProgressBar

import egamma_tnp
from egamma_tnp import ElectronTagNProbeFromNanoAOD


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def assert_arrays_equal(a1, a2):
    for i in a1.fields:
        assert ak.all(a1[i] == a2[i])
    for j in a2.fields:
        assert ak.all(a1[j] == a2[j])


def test_cli():
    subprocess.run(
        "run_analysis --config tests/example_runner.json --settings tests/example_settings.json --fileset tests/example_fileset.json --binning tests/example_binning.json --output tests/output --executor threads",
        shell=True,
        check=True,
    )

    with open("tests/example_fileset.json") as f:
        fileset = json.load(f)

    egamma_tnp.binning.set("el_eta_bins", [-2.5, -2.0, -1.566, -1.4442, -1.0, 0.0, 1.0, 1.4442, 1.566, 2.0, 2.5])

    workflow = ElectronTagNProbeFromNanoAOD(
        fileset=fileset,
        filters=["HLT_Ele30_WPTight_Gsf", "cutBased >= 2"],
        filterbit=[1, None],
        trigger_pt=[30, None],
        tags_pt_cut=35,
        probes_pt_cut=27,
        tags_abseta_cut=2.17,
        probes_abseta_cut=2.5,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args={},
        use_sc_eta=True,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        require_event_to_pass_hlt_filter=True,
    )

    get_tnp_arrays_1 = workflow.get_tnp_arrays(
        cut_and_count=False,
        mass_range=None,
        vars="all",
        flat=True,
        uproot_options={"allow_read_errors_with_report": True, "timeout": 120},
    )
    get_tnp_arrays_2 = workflow.get_tnp_arrays(
        cut_and_count=True,
        mass_range=None,
        vars=["el_pt", "el_eta", "Jet_pt", "MET_sumEt"],
        flat=False,
        uproot_options={"allow_read_errors_with_report": (OSError, ValueError)},
    )
    get_passing_and_failing_probes_1_hlt = workflow.get_passing_and_failing_probes(
        filter="HLT_Ele30_WPTight_Gsf",
        mass_range=None,
        cut_and_count=True,
        vars="all",
        flat=True,
        uproot_options=None,
    )
    get_1d_pt_eta_phi_tnp_histograms_1_hlt = workflow.get_1d_pt_eta_phi_tnp_histograms(
        filter="HLT_Ele30_WPTight_Gsf",
        cut_and_count=False,
        mass_range=None,
        plateau_cut=None,
        eta_regions_pt=None,
        phi_regions_eta=None,
        eta_regions_phi=None,
        vars=["el_pt", "el_eta", "el_phi"],
        uproot_options=None,
    )
    get_nd_tnp_histograms_1_hlt = workflow.get_nd_tnp_histograms(
        filter="HLT_Ele30_WPTight_Gsf",
        cut_and_count=False,
        mass_range=None,
        vars=["el_pt", "el_eta", "el_phi"],
        uproot_options=None,
    )
    get_nd_tnp_histograms_1_id = workflow.get_nd_tnp_histograms(
        filter="cutBased >= 2",
        cut_and_count=False,
        mass_range=None,
        vars=["el_pt", "el_eta", "el_phi"],
        uproot_options=None,
    )

    to_compute = {
        "get_tnp_arrays_1": get_tnp_arrays_1,
        "get_tnp_arrays_2": get_tnp_arrays_2,
        "get_passing_and_failing_probes_1_hlt": get_passing_and_failing_probes_1_hlt,
        "get_1d_pt_eta_phi_tnp_histograms_1_hlt": get_1d_pt_eta_phi_tnp_histograms_1_hlt,
        "get_nd_tnp_histograms_1_hlt": get_nd_tnp_histograms_1_hlt,
        "get_nd_tnp_histograms_1_id": get_nd_tnp_histograms_1_id,
    }

    with ProgressBar():
        (out,) = dask.compute(to_compute)

    assert_arrays_equal(
        out["get_tnp_arrays_1"][0]["sample/1"],
        ak.from_parquet("tests/output/sample_1/get_tnp_arrays_1/NTuples-part0.parquet"),
    )
    assert os.path.exists("tests/output/sample_1/get_tnp_arrays_1/report.json")
    assert_arrays_equal(
        out["get_tnp_arrays_1"][0]["sample/2"],
        ak.from_parquet("tests/output/sample_2/get_tnp_arrays_1/NTuples-part0.parquet"),
    )
    assert os.path.exists("tests/output/sample_2/get_tnp_arrays_1/report.json")

    assert_arrays_equal(
        out["get_tnp_arrays_2"][0]["sample/1"],
        ak.from_parquet("tests/output/sample_1/get_tnp_arrays_2/NTuples-part0.parquet"),
    )
    assert os.path.exists("tests/output/sample_1/get_tnp_arrays_2/report.json")
    assert_arrays_equal(
        out["get_tnp_arrays_2"][0]["sample/2"],
        ak.from_parquet("tests/output/sample_2/get_tnp_arrays_2/NTuples-part0.parquet"),
    )
    assert os.path.exists("tests/output/sample_2/get_tnp_arrays_2/report.json")

    assert_arrays_equal(
        out["get_passing_and_failing_probes_1_hlt"]["sample/1"]["passing"],
        ak.from_parquet("tests/output/sample_1/get_passing_and_failing_probes_1/passing_HLT_Ele30_WPTight_Gsf_NTuples-part0.parquet"),
    )
    assert_arrays_equal(
        out["get_passing_and_failing_probes_1_hlt"]["sample/1"]["failing"],
        ak.from_parquet("tests/output/sample_1/get_passing_and_failing_probes_1/failing_HLT_Ele30_WPTight_Gsf_NTuples-part0.parquet"),
    )
    assert_arrays_equal(
        out["get_passing_and_failing_probes_1_hlt"]["sample/2"]["passing"],
        ak.from_parquet("tests/output/sample_2/get_passing_and_failing_probes_1/passing_HLT_Ele30_WPTight_Gsf_NTuples-part0.parquet"),
    )
    assert_arrays_equal(
        out["get_passing_and_failing_probes_1_hlt"]["sample/2"]["failing"],
        ak.from_parquet("tests/output/sample_2/get_passing_and_failing_probes_1/failing_HLT_Ele30_WPTight_Gsf_NTuples-part0.parquet"),
    )

    with open("tests/output/sample_1/get_1d_pt_eta_phi_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["pt"]["barrel"]["passing"],
        histos["pt"]["barrel"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["pt"]["barrel"]["failing"],
        histos["pt"]["barrel"]["failing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["eta"]["entire"]["passing"],
        histos["eta"]["entire"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["eta"]["entire"]["failing"],
        histos["eta"]["entire"]["failing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["phi"]["entire"]["passing"],
        histos["phi"]["entire"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/1"]["phi"]["entire"]["failing"],
        histos["phi"]["entire"]["failing"],
        flow=True,
    )

    with open("tests/output/sample_2/get_1d_pt_eta_phi_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["pt"]["barrel"]["passing"],
        histos["pt"]["barrel"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["pt"]["barrel"]["failing"],
        histos["pt"]["barrel"]["failing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["eta"]["entire"]["passing"],
        histos["eta"]["entire"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["eta"]["entire"]["failing"],
        histos["eta"]["entire"]["failing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["phi"]["entire"]["passing"],
        histos["phi"]["entire"]["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_1d_pt_eta_phi_tnp_histograms_1_hlt"]["sample/2"]["phi"]["entire"]["failing"],
        histos["phi"]["entire"]["failing"],
        flow=True,
    )

    with open("tests/output/sample_1/get_nd_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_hlt"]["sample/1"]["passing"],
        histos["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_hlt"]["sample/1"]["failing"],
        histos["failing"],
        flow=True,
    )
    assert os.path.exists("tests/output/sample_1/get_nd_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_report.json")
    with open("tests/output/sample_1/get_nd_tnp_histograms_1/cutBased_gte_2_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_id"]["sample/1"]["passing"],
        histos["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_id"]["sample/1"]["failing"],
        histos["failing"],
        flow=True,
    )
    assert os.path.exists("tests/output/sample_1/get_nd_tnp_histograms_1/cutBased_gte_2_report.json")

    with open("tests/output/sample_2/get_nd_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_hlt"]["sample/2"]["passing"],
        histos["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_hlt"]["sample/2"]["failing"],
        histos["failing"],
        flow=True,
    )
    assert os.path.exists("tests/output/sample_2/get_nd_tnp_histograms_1/HLT_Ele30_WPTight_Gsf_report.json")
    with open("tests/output/sample_2/get_nd_tnp_histograms_1/cutBased_gte_2_histos.pkl", "rb") as f:
        histos = pickle.load(f)
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_id"]["sample/2"]["passing"],
        histos["passing"],
        flow=True,
    )
    assert_histograms_equal(
        out["get_nd_tnp_histograms_1_id"]["sample/2"]["failing"],
        histos["failing"],
        flow=True,
    )
    assert os.path.exists("tests/output/sample_2/get_nd_tnp_histograms_1/cutBased_gte_2_report.json")
