from __future__ import annotations

from egamma_tnp.utils.dataset import redirect_files
from egamma_tnp.utils.histogramming import (
    convert_2d_mll_hists_to_1d_hists,
    convert_nd_mll_hists_to_1d_hists,
    create_hists_root_file_for_fitter,
    fill_nd_cutncount_histograms,
    fill_nd_mll_histograms,
    fill_pt_eta_phi_cutncount_histograms,
    fill_pt_eta_phi_mll_histograms,
    get_ratio_histogram,
)
from egamma_tnp.utils.misc import (
    calculate_photon_SC_eta,
    calculate_photon_SC_eta_numpy,
    custom_delta_r,
    dask_calculate_photon_SC_eta,
    delta_r_SC,
)

__all__ = (
    "redirect_files",
    "fill_pt_eta_phi_cutncount_histograms",
    "fill_pt_eta_phi_mll_histograms",
    "fill_nd_cutncount_histograms",
    "fill_nd_mll_histograms",
    "convert_2d_mll_hists_to_1d_hists",
    "convert_nd_mll_hists_to_1d_hists",
    "create_hists_root_file_for_fitter",
    "get_ratio_histogram",
    "delta_r_SC",
    "custom_delta_r",
    "calculate_photon_SC_eta",
    "calculate_photon_SC_eta_numpy",
    "dask_calculate_photon_SC_eta",
)


def dir():
    return __all__
