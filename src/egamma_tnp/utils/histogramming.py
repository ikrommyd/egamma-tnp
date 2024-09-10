from __future__ import annotations

import awkward as ak
import fsspec
import numpy as np
import uproot
from hist import intervals


def flatten_array(array):
    """Flatten the probes array.

    Parameters
    ----------
        array : awkward.Array or dask_awkward.Array
            An array with the fields to be flattened.

    Returns
    -------
        awkward.Array or dask_awkward.Array
            The flattened array.
    """

    return ak.flatten(ak.zip({var: array[var] for var in array.fields}), axis=-1)


def get_ratio_histogram(passing_probes, failing_or_all_probes, denominator_type="failing"):
    """Get the ratio (efficiency) of the passing over passing + failing probes.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        failing_or_all_probes : hist.Hist
            The histogram of the failing or passing + failing probes.
        denominator_type : str, optional
            The type of the denominator histogram.
            Can be either "failing" or "all".
            The default is "failing".

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    import hist

    if passing_probes.axes != failing_or_all_probes.axes:
        raise ValueError("The axes of the histograms must be the same.")
    if denominator_type == "failing":
        all_probes = passing_probes + failing_or_all_probes
    elif denominator_type == "all":
        all_probes = failing_or_all_probes
    else:
        raise ValueError("Invalid denominator type. Must be either 'failing' or 'all'.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = passing_probes.values(flow=True) / all_probes.values(flow=True)
    ratio = hist.Hist(hist.Hist(*passing_probes.axes))
    ratio[:] = np.nan_to_num(ratio_values)
    yerr = intervals.ratio_uncertainty(passing_probes.values(), all_probes.values(), uncertainty_type="efficiency")

    return ratio, yerr


def fill_pt_eta_phi_cutncount_histograms(
    passing_probes,
    failing_probes,
    plateau_cut=None,
    eta_regions_pt=None,
    phi_regions_eta=None,
    eta_regions_phi=None,
    vars=None,
):
    """Get the Pt, Eta and Phi histograms of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the failing probes.
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions_pt : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Pt histograms will be split into those eta regions.
            The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
        phi_regions_eta : dict, optional
            A dictionary of the form `{"name": [phimin, phimax], ...}`
            where name is the name of the region and phimin and phimax are the absolute phi bounds.
            The Eta histograms will be split into those phi regions.
            The default is to use the entire |phi| < 2.5 region.
        eta_regions_phi : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Phi histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        vars : list, optional
            A list of the fields that refer to the Pt, Eta, and Phi of the probes.
            Must be in the order of Pt, Eta, and Phi.
            The default is ["el_pt", "el_eta", "el_phi"].

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"var": {"name": {"passing": passing_probes, "failing": failing_probes}, ...}, ...}`
            where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
            Each `"name"` is the name of eta region specified by the user.
            `passing_probes` and `failing_probes` are `hist.Hist` or `hist.dask.Hist` objects.
            These are the histograms of the passing and failing probes respectively.
    """
    import hist

    if isinstance(passing_probes, ak.Array) and isinstance(failing_probes, ak.Array):
        from hist import Hist
    else:
        from hist.dask import Hist

    import egamma_tnp

    if "weight" not in passing_probes.fields or "weight" not in failing_probes.fields:
        passing_probes["weight"] = 1
        failing_probes["weight"] = 1
    passing_probes, failing_probes = flatten_array(passing_probes), flatten_array(failing_probes)

    if plateau_cut is None:
        plateau_cut = 0
    if eta_regions_pt is None:
        eta_regions_pt = {
            "barrel": [0.0, 1.4442],
            "endcap": [1.566, 2.5],
        }
    if phi_regions_eta is None:
        phi_regions_eta = {"entire": [0.0, 3.32]}
    if eta_regions_phi is None:
        eta_regions_phi = {"entire": [0.0, 2.5]}
    if vars is None:
        vars = ["el_pt", "el_eta", "el_phi"]

    if any(egamma_tnp.binning.get(f"{var}_bins") is None for var in vars):
        raise ValueError(
            """One or more variables do not have binning information.
            Please define the binning information using `egamma_tnp.binning.set`.
            The variable names in the configuration json should be in the form of `"{var}_bins"`."""
        )
    ptbins = egamma_tnp.binning.get(f"{vars[0]}_bins")
    etabins = egamma_tnp.binning.get(f"{vars[1]}_bins")
    phibins = egamma_tnp.binning.get(f"{vars[2]}_bins")

    pt_pass = passing_probes[vars[0]]
    pt_fail = failing_probes[vars[0]]
    eta_pass = passing_probes[vars[1]]
    eta_fail = failing_probes[vars[1]]
    phi_pass = passing_probes[vars[2]]
    phi_fail = failing_probes[vars[2]]
    passing_probes_weight = passing_probes.weight
    failing_probes_weight = failing_probes.weight

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass = pt_pass > plateau_cut
    plateau_mask_fail = pt_fail > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (abs(eta_pass) < region_pt[1])
        eta_mask_pt_fail = (abs(eta_fail) > region_pt[0]) & (abs(eta_fail) < region_pt[1])
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_fail = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_pass.fill(pt_pass[eta_mask_pt_pass], weight=passing_probes_weight[eta_mask_pt_pass])
        hpt_fail.fill(pt_fail[eta_mask_pt_fail], weight=failing_probes_weight[eta_mask_pt_fail])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "failing": hpt_fail}

    for name_eta, region_eta in phi_regions_eta.items():
        phi_mask_eta_pass = (abs(phi_pass) > region_eta[0]) & (abs(phi_pass) < region_eta[1])
        phi_mask_eta_fail = (abs(phi_fail) > region_eta[0]) & (abs(phi_fail) < region_eta[1])
        heta_pass = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            storage=hist.storage.Weight(),
        )
        heta_fail = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            storage=hist.storage.Weight(),
        )
        heta_pass.fill(eta_pass[plateau_mask_pass & phi_mask_eta_pass], weight=passing_probes_weight[plateau_mask_pass & phi_mask_eta_pass])
        heta_fail.fill(eta_fail[plateau_mask_fail & phi_mask_eta_fail], weight=failing_probes_weight[plateau_mask_fail & phi_mask_eta_fail])

        histograms["eta"][name_eta] = {"passing": heta_pass, "failing": heta_fail}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (abs(eta_pass) < region_phi[1])
        eta_mask_phi_fail = (abs(eta_fail) > region_phi[0]) & (abs(eta_fail) < region_phi[1])
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            storage=hist.storage.Weight(),
        )
        hphi_fail = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            storage=hist.storage.Weight(),
        )
        hphi_pass.fill(phi_pass[plateau_mask_pass & eta_mask_phi_pass], weight=passing_probes_weight[plateau_mask_pass & eta_mask_phi_pass])
        hphi_fail.fill(phi_fail[plateau_mask_fail & eta_mask_phi_fail], weight=failing_probes_weight[plateau_mask_fail & eta_mask_phi_fail])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "failing": hphi_fail}

    return histograms


def fill_pt_eta_phi_mll_histograms(
    passing_probes,
    failing_probes,
    plateau_cut=None,
    eta_regions_pt=None,
    phi_regions_eta=None,
    eta_regions_phi=None,
    vars=None,
):
    """Get the 2D histograms of Pt, Eta and Phi vs mll of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the failing probes.
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions_pt : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Pt histograms will be split into those eta regions.
            The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
        phi_regions_eta : dict, optional
            A dictionary of the form `{"name": [phimin, phimax], ...}`
            where name is the name of the region and phimin and phimax are the absolute phi bounds.
            The Eta histograms will be split into those phi regions.
            The default is to use the entire |phi| < 2.5 region.
        eta_regions_phi : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Phi histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        vars : list, optional
            A list of the fields that refer to the Pt, Eta, and Phi of the probes.
            Must be in the order of Pt, Eta, and Phi.
            The default is ["el_pt", "el_eta", "el_phi"].

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"var": {"name": {"passing": passing_probes, "failing": failing_probes}, ...}, ...}`
            where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
            Each `"name"` is the name of eta region specified by the user.
            `passing_probes` and `failing_probes` are `hist.Hist` or `hist.dask.Hist` objects.
            These are the histograms of the passing and failing probes respectively.
    """
    import hist

    if isinstance(passing_probes, ak.Array) and isinstance(failing_probes, ak.Array):
        from hist import Hist
    else:
        from hist.dask import Hist

    import egamma_tnp

    if "weight" not in passing_probes.fields or "weight" not in failing_probes.fields:
        passing_probes["weight"] = 1
        failing_probes["weight"] = 1
    passing_probes, failing_probes = flatten_array(passing_probes), flatten_array(failing_probes)

    if plateau_cut is None:
        plateau_cut = 0
    if eta_regions_pt is None:
        eta_regions_pt = {
            "barrel": [0.0, 1.4442],
            "endcap": [1.566, 2.5],
        }
    if phi_regions_eta is None:
        phi_regions_eta = {"entire": [0.0, 3.32]}
    if eta_regions_phi is None:
        eta_regions_phi = {"entire": [0.0, 2.5]}
    if vars is None:
        vars = ["el_pt", "el_eta", "el_phi"]

    if any(egamma_tnp.binning.get(f"{var}_bins") is None for var in vars):
        raise ValueError(
            """One or more variables do not have binning information.
            Please define the binning information using `egamma_tnp.binning.set`.
            The variable names in the configuration json should be in the form of `"{var}_bins"`."""
        )
    ptbins = egamma_tnp.binning.get(f"{vars[0]}_bins")
    etabins = egamma_tnp.binning.get(f"{vars[1]}_bins")
    phibins = egamma_tnp.binning.get(f"{vars[2]}_bins")

    pt_pass = passing_probes[vars[0]]
    pt_fail = failing_probes[vars[0]]
    eta_pass = passing_probes[vars[1]]
    eta_fail = failing_probes[vars[1]]
    phi_pass = passing_probes[vars[2]]
    phi_fail = failing_probes[vars[2]]
    mll_pass = passing_probes.pair_mass
    mll_fail = failing_probes.pair_mass
    passing_probes_weight = passing_probes.weight
    failing_probes_weight = failing_probes.weight

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass = pt_pass > plateau_cut
    plateau_mask_fail = pt_fail > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (abs(eta_pass) < region_pt[1])
        eta_mask_pt_fail = (abs(eta_fail) > region_pt[0]) & (abs(eta_fail) < region_pt[1])
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_fail = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_pass.fill(pt_pass[eta_mask_pt_pass], mll_pass[eta_mask_pt_pass], weight=passing_probes_weight[eta_mask_pt_pass])
        hpt_fail.fill(pt_fail[eta_mask_pt_fail], mll_fail[eta_mask_pt_fail], weight=failing_probes_weight[eta_mask_pt_fail])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "failing": hpt_fail}

    for name_eta, region_eta in phi_regions_eta.items():
        phi_mask_eta_pass = (abs(phi_pass) > region_eta[0]) & (abs(phi_pass) < region_eta[1])
        phi_mask_eta_fail = (abs(phi_fail) > region_eta[0]) & (abs(phi_fail) < region_eta[1])
        heta_pass = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        heta_fail = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        eta_mask_fail = plateau_mask_fail & phi_mask_eta_fail
        eta_mask_pass = plateau_mask_pass & phi_mask_eta_pass
        heta_pass.fill(eta_pass[eta_mask_pass], mll_pass[eta_mask_pass], weight=passing_probes_weight[eta_mask_pass])
        heta_fail.fill(eta_fail[eta_mask_fail], mll_fail[eta_mask_fail], weight=failing_probes_weight[eta_mask_fail])

        histograms["eta"][name_eta] = {"passing": heta_pass, "failing": heta_fail}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (abs(eta_pass) < region_phi[1])
        eta_mask_phi_fail = (abs(eta_fail) > region_phi[0]) & (abs(eta_fail) < region_phi[1])
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hphi_fail = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        phi_mask_pass = plateau_mask_pass & eta_mask_phi_pass
        phi_mask_fail = plateau_mask_fail & eta_mask_phi_fail
        hphi_pass.fill(phi_pass[phi_mask_pass], mll_pass[phi_mask_pass], weight=passing_probes_weight[phi_mask_pass])
        hphi_fail.fill(phi_fail[phi_mask_fail], mll_fail[phi_mask_fail], weight=failing_probes_weight[phi_mask_fail])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "failing": hphi_fail}

    return histograms


def fill_nd_cutncount_histograms(
    passing_probes,
    failing_probes,
    vars=None,
):
    """
    Get the N-dimensional histogram of the passing and failing probes.
    The histogram will have axes for each variable specified in `vars`.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the failing probes.
        vars : list, optional
            A list of the fields to use as axes in the histogram.
            The default is ["el_pt", "el_eta", "el_phi"].

    Returns
    -------
        histograms : dict
        A dictionary of the form {"passing": hpass, "failing": hfail} where
        hpass : hist.Hist or hist.dask.Hist
            An N-dimensional histogram of the passing probes.
        hfail : hist.Hist
            An N-dimensional histogram of the failing probes.
    """
    if vars is None:
        vars = ["el_pt", "el_eta", "el_phi"]
    if isinstance(vars, str):
        raise ValueError("Please provide a list of variables and not a single string.")

    import hist

    if isinstance(passing_probes, ak.Array) and isinstance(failing_probes, ak.Array):
        from hist import Hist
    else:
        from hist.dask import Hist

    import egamma_tnp

    if "weight" not in passing_probes.fields or "weight" not in failing_probes.fields:
        passing_probes["weight"] = 1
        failing_probes["weight"] = 1
    passing_probes, failing_probes = flatten_array(passing_probes), flatten_array(failing_probes)

    if any(egamma_tnp.binning.get(f"{var}_bins") is None for var in vars):
        raise ValueError(
            """One or more variables do not have binning information.
            Please define the binning information using `egamma_tnp.binning.set`.
            The variable names in the configuration json should be in the form of `"{var}_bins"`."""
        )

    axes = [hist.axis.Variable(egamma_tnp.binning.get(f"{var}_bins"), name=var, label=f"{var.capitalize()}") for var in vars]

    hpass = Hist(*axes, storage=hist.storage.Weight())
    hfail = Hist(*axes, storage=hist.storage.Weight())

    hpass.fill(*[passing_probes[var] for var in vars], weight=passing_probes.weight)
    hfail.fill(*[failing_probes[var] for var in vars], weight=failing_probes.weight)

    return {"passing": hpass, "failing": hfail}


def fill_nd_mll_histograms(
    passing_probes,
    failing_probes,
    vars=None,
):
    """
    Get the N+1-dimensional histogram of the passing and failing probes.
    The histogram will have axes for each variable specified in `vars` and an invariant mass axis.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with the fields specified in `vars` of the failing probes.
        vars : list, optional
            A list of the fields to use as axes in the histogram.
            The default is ["el_pt", "el_eta", "el_phi"].

    Returns
    -------
        histograms : dict
        A dictionary of the form {"passing": hpass, "failing": hfail} where
        hpass : hist.Hist or hist.dask.Hist
            An N+1-dimensional histogram of the passing probes.
        hfail : hist.Hist
            An N+1-dimensional histogram of the failing probes.
    """
    if vars is None:
        vars = ["el_pt", "el_eta", "el_phi"]
    if isinstance(vars, str):
        raise ValueError("Please provide a list of variables and not a single string.")

    import hist

    if isinstance(passing_probes, ak.Array) and isinstance(failing_probes, ak.Array):
        from hist import Hist
    else:
        from hist.dask import Hist

    import egamma_tnp

    if "weight" not in passing_probes.fields or "weight" not in failing_probes.fields:
        passing_probes["weight"] = 1
        failing_probes["weight"] = 1
    passing_probes, failing_probes = flatten_array(passing_probes), flatten_array(failing_probes)

    if any(egamma_tnp.binning.get(f"{var}_bins") is None for var in vars):
        raise ValueError(
            """One or more variables do not have binning information.
            Please define the binning information using `egamma_tnp.binning.set`.
            The variable names in the configuration json should be in the form of `"{var}_bins"`."""
        )

    axes = [hist.axis.Variable(egamma_tnp.binning.get(f"{var}_bins"), name=var, label=f"{var.capitalize()}") for var in vars]
    axes.append(hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"))

    hpass = Hist(*axes, storage=hist.storage.Weight())
    hfail = Hist(*axes, storage=hist.storage.Weight())

    hpass.fill(*[passing_probes[var] for var in vars], passing_probes.pair_mass, weight=passing_probes.weight)
    hfail.fill(*[failing_probes[var] for var in vars], failing_probes.pair_mass, weight=failing_probes.weight)

    return {"passing": hpass, "failing": hfail}


# Function to convert edge value to string format for keys
def _format_edge(value):
    prefix = "m" if value < 0 else ""
    formatted = f"{prefix}{abs(value):.2f}".replace(".", "p")
    return formatted


# We could have used the function right below to convert 2D histograms to 1D histograms
# with just using axes=[the name of the x axis] but I prefer this easier approach for now.
# It makes debugging easier and it is more explicit
def _convert_2d_mll_hist_to_1d_hists(h2d):
    histograms = {}
    bin_info_list = []
    ax = h2d.axes.name[0]

    for idx in range(h2d.axes[0].size):
        min_edge = h2d.axes[0].edges[idx]
        max_edge = h2d.axes[0].edges[idx + 1]
        key = f"{ax}_{_format_edge(min_edge)}To{_format_edge(max_edge)}"
        histograms[key] = h2d[{ax: idx}]
        bin_name = f"bin{idx}_{key}"
        bin_info_list.append(
            {
                "cut": f"{ax} >= {min_edge:.6f} && {ax} < {max_edge:.6f}",
                "name": bin_name,
                "vars": {ax: {"min": min_edge, "max": max_edge}},
                "title": f"; {min_edge:.3f} < {ax} < {max_edge:.3f}",
            }
        )

    binning = {"bins": bin_info_list, "vars": [ax]}

    return histograms, binning


def _convert_nd_mll_hist_to_1d_hists(h4d, axes):
    import itertools

    import hist

    # Initialize the slicer to keep specified axes and sum over the others except 'mll'
    s = hist.tag.Slicer()
    slice_dict = {ax.name: s[:] if ax.name in axes or ax.name == "mll" else s[sum] for ax in h4d.axes}

    # Apply slicing to obtain the relevant projection while keeping 'mll' intact
    h = h4d[slice_dict]

    # Dictionary to hold all 1D histograms
    histograms = {}
    # List to hold all dictionaries for each bin
    bin_info_list = []

    # Generate all combinations of specified axes except 'mll'
    total_bins = np.prod([h.axes[ax].size for ax in axes])
    zfill_length = len(str(total_bins))
    counter = 0
    axes_reversed = axes[::-1]
    for reversed_idx_combination in itertools.product(*(range(h.axes[ax].size) for ax in axes_reversed)):
        idx_combination = reversed_idx_combination[::-1]
        bin_details = {}
        vars_details = {}
        cut_parts = []
        title_parts = []

        # Construct details using the given order
        for ax, idx in zip(axes, idx_combination):
            min_edge = h.axes[ax].edges[idx]
            max_edge = h.axes[ax].edges[idx + 1]
            vars_details[ax] = {"min": min_edge, "max": max_edge}
            cut_parts.append(f"{ax} >= {min_edge:.6f} && {ax} < {max_edge:.6f}")
            title_parts.append(f"{min_edge:.3f} < {ax} < {max_edge:.3f}")

        # Key should be constructed in the order given using the _format_edge for key consistency
        key = "_".join(f"{ax}_{_format_edge(vars_details[ax]['min'])}To{_format_edge(vars_details[ax]['max'])}" for ax in axes)

        # Correcting slice indices using original axis ordering
        slice_indices = {ax: idx_combination[axes.index(ax)] for ax in axes}
        slice_indices["mll"] = slice(None)
        histograms[key] = h[slice_indices]

        # Create the dictionary for this bin
        bin_name = f"bin{str(counter).zfill(zfill_length)}_{key}"
        bin_details["cut"] = " && ".join(cut_parts)
        bin_details["name"] = bin_name
        bin_details["vars"] = vars_details
        bin_details["title"] = "; " + "; ".join(title_parts)

        bin_info_list.append(bin_details)
        counter += 1

    binning = {"bins": bin_info_list, "vars": axes}

    return histograms, binning


def convert_2d_mll_hists_to_1d_hists(hist_dict):
    """Convert 2D (var, mll) histogram dict to 1D histograms.
    This will create a 1D histogram for each bin in the x-axis of the 2D histograms.

    Parameters
    ----------
        hist_dict : dict
            A dictionary of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
            where each hist.Hist is a 2D histogram with axes (var, mll).
    Returns
    -------
        histograms : dict
            A dictionary of the form {"var": {"region": {"passing": {bin_name: hist.Hist, ...}, "failing": {bin_name: hist.Hist, ...}, "binning": binning}, ...}, ...}
    """
    histograms = {}  # Create a new dictionary instead of modifying the original
    for var, region_dict in hist_dict.items():
        histograms[var] = {}  # Initialize var dictionary
        for region_name, hists in region_dict.items():
            histograms[var][region_name] = {}  # Initialize region dictionary
            for histname, h in hists.items():
                if h.ndim != 2:
                    raise ValueError("Input histograms must be 2D.")
                hs, binning = _convert_2d_mll_hist_to_1d_hists(h)
                histograms[var][region_name][histname] = hs  # Populate with new histograms
                histograms[var][region_name]["binning"] = binning  # Set binning for this region
    return histograms


def convert_nd_mll_hists_to_1d_hists(hists, axes=None):
    """Convert N+1 dimensional (axes, mll) histograms to 1D histograms.
    This will create a 1D histogram for each combination of bins in the specified axes.
    For instance, if axes are ["pt, "eta", "phi"] and if you have `npt` bins in Pt, `neta` bins in Eta, and `nphi` bins in Phi,
    then you will get `npt x neta x nphi` 1D histograms.
    If a variable is not specified in the axes, then it will be summed over.
    For instance if you specify only `pt` and `eta`, then the 1D histograms will be
    created for each pair of Pt and Eta bins, summed over all Phi bins, so `npt x neta` histograms.

    Parameters
    ----------
        hists : dict
            A dictionary of the form {"passing": hpass, "failing": hfail}
            where hpass and hfail are 4D histograms with axes (Pt, Eta, Phi, mll).
        axes : list, optional
            A list of the axes to keep in the 1D histograms.
            The default is ["el_pt", "el_eta"].

    Returns
    -------
        histograms : dict
            A dictionary of the form {"passing": hpass, "failing": hfail}
            where hpass and hfail are dictionaries of 1D histograms.
            The keys are the bin combinations of the specified axes
            and the values are the 1D histograms for each bin combination.
        binning : dict
            A dictionary with the binning information.
    """
    if axes is None:
        axes = ["el_pt", "el_eta"]
    if len(set(axes)) != len(axes):
        raise ValueError("All axes must be unique.")

    histograms = {}
    for key, h4d in hists.items():
        if h4d.axes[-1].name != "mll":
            raise ValueError("The last axis must be an invariant mass axis.")
        hists, binning = _convert_nd_mll_hist_to_1d_hists(h4d, axes=axes)
        histograms[key] = hists

    return histograms, binning


def create_hists_root_file_for_fitter(hists, root_path, binning_path, axes=None):
    """Create a ROOT file with 1D histograms of passing and failing probes.
    To be used as input to the fitter.

    Parameters
    ----------
        hists : dict
            Either a dictionary of 2D histograms of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
            or a dictionary of 4D histograms of the form {"passing": hpass, "failing": hfail}.
            where hpass and hfail are 4D histograms with axes (Pt, Eta, Phi, mll).
        root_path : str
            The path to the ROOT file.
        binning_path : str
            The path to the pickle file with the binning information.
        axes : list, optional
            A list of the axes to keep in the 1D histograms.
            The default is ["el_pt", "el_eta"].

        Notes
        -----
            If the input is a dictionary of 2D histograms, then multiple ROOT files and binning files will be created,
            one for each variable and region present in the input.
            For each variable and region present, a trailing `_<var>_<region>.root` and `_<var>_<region>.pkl` will be added to the root_path and binning_path respectively.
    """
    import pickle

    import uproot

    if axes is None:
        axes = ["el_pt", "el_eta"]
    if len(set(axes)) != len(axes):
        raise ValueError("All axes must be unique.")

    if isinstance(hists, dict) and "passing" in hists and "failing" in hists:
        histograms, binning = convert_nd_mll_hists_to_1d_hists(hists, axes=axes)
        passing_hists = histograms["passing"]
        failing_hists = histograms["failing"]

        if passing_hists.keys() != failing_hists.keys():
            raise ValueError("Passing and failing histograms must have the same binning.")

        names = list(passing_hists.keys())
        max_number = len(str(len(names)))

        with uproot.recreate(root_path) as f:
            counter = 0
            for name in names:
                counter_str = str(counter).zfill(max_number)
                f[f"bin{counter_str}_{name}_Pass"] = passing_hists[name]
                f[f"bin{counter_str}_{name}_Fail"] = failing_hists[name]
                counter += 1

        with fsspec.open(binning_path, "wb") as f:
            pickle.dump(binning, f, protocol=2)

    elif isinstance(hists, dict) and "pt" in hists and "eta" in hists and "phi" in hists:
        histograms = convert_2d_mll_hists_to_1d_hists(hists)
        for var, region_dict in histograms.items():
            for region_name, hists in region_dict.items():
                new_path = root_path.replace(".root", f"_{var}_{region_name}.root")
                new_binning_path = binning_path.replace(".pkl", f"_{var}_{region_name}.pkl")
                with uproot.recreate(new_path) as f:
                    passing_hists = hists["passing"]
                    failing_hists = hists["failing"]
                    names = list(passing_hists.keys())
                    max_number = len(str(len(names)))
                    counter = 0
                    for name in names:
                        counter_str = str(counter).zfill(max_number)
                        f[f"bin{counter_str}_{name}_Pass"] = passing_hists[name]
                        f[f"bin{counter_str}_{name}_Fail"] = failing_hists[name]
                        counter += 1

                with fsspec.open(new_binning_path, "wb") as f:
                    pickle.dump(hists["binning"], f, protocol=2)

    else:
        raise ValueError("Invalid `hists` format")


def save_hists(path, res):
    """Save histograms to a ROOT file.

    Parameters
    ----------
        path : str
            The path to the ROOT file.
        res : dict
            A histogram dictionary of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
    """
    with uproot.recreate(path) as f:
        for var, region_dict in res.items():
            for region_name, hists in region_dict.items():
                for histname, h in hists.items():
                    f[f"{var}/{region_name}/{histname}"] = h
