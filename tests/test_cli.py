from __future__ import annotations

import json
import subprocess

import dask
import numpy as np
from dask.diagnostics import ProgressBar

from egamma_tnp import ElectronTagNProbeFromNanoAOD


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def test_cli():
    subprocess.run(
        "run_analysis --config tests/example_runner.json --settings tests/example_settings.json --fileset tests/example_fileset.json --binning tests/example_binning.json --output tests/output --executor threads",
        shell=True,
        check=True,
    )
    with open("tests/example_fileset.json") as f:
        fileset = json.load(f)

    workflow = ElectronTagNProbeFromNanoAOD(
        fileset=fileset,
        filters=["HLT_Ele30_WPTight_Gsf", "cutBased >= 2"],
        filterbit=[1, None],
        trigger_pt=[30, None],
    )

    get_tnp_arrays_1 = workflow.get_tnp_arrays(
        cut_and_count=False, mass_range=None, vars="all", flat=True, uproot_options={"allow_read_errors_with_report": True, "timeout": 120}
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
    get_passing_and_failing_probes_1_id = workflow.get_passing_and_failing_probes(
        filter="cutBased >= 2",
        mass_range=None,
        cut_and_count=True,
        vars="all",
        flat=True,
        uproot_options=None,
    )
    get_1d_pt_eta_phi_tnp_histograms_1_hlt = workflow.get_1d_pt_eta_phi_tnp_histograms(
        filter="HLT_Ele30_WPTight_Gsf",
        cut_and_count=True,
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
        cut_and_count=True,
        mass_range=None,
        vars=["el_pt", "el_eta", "el_phi"],
        uproot_options=None,
    )
    get_nd_tnp_histograms_1_id = workflow.get_nd_tnp_histograms(
        filter="cutBased >= 2",
        cut_and_count=True,
        mass_range=None,
        vars=["el_pt", "el_eta", "el_phi"],
        uproot_options=None,
    )

    with ProgressBar():
        (out,) = dask.compute(
            [
                get_tnp_arrays_1,
                get_tnp_arrays_2,
                get_passing_and_failing_probes_1_hlt,
                get_passing_and_failing_probes_1_id,
                get_1d_pt_eta_phi_tnp_histograms_1_hlt,
                get_nd_tnp_histograms_1_hlt,
                get_nd_tnp_histograms_1_id,
            ]
        )
