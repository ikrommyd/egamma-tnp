import numpy as np
import uproot
from hist import intervals


def get_ratio_histogram(passing_probes, all_probes):
    """Get the ratio (efficiency) of the passing and all probes histograms.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        all_probes : hist.Hist
            The histogram of all probes.

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    ratio = passing_probes / all_probes
    ratio[:] = np.nan_to_num(ratio.values())
    yerr = intervals.ratio_uncertainty(
        passing_probes.values(), all_probes.values(), uncertainty_type="efficiency"
    )

    return ratio, yerr


def fill_tnp_histograms(
    passing_probes,
    all_probes,
    plateau_cut=None,
    eta_regions_pt=None,
    eta_regions_eta=None,
    eta_regions_phi=None,
    delayed=True,
):
    """Get the Pt and Eta histograms of the passing and all probes.

    Parameters
    ----------
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions_pt : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Pt histograms will be split into those eta regions.
            The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
        eta_regions_eta : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Eta histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        eta_regions_phi : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Phi histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        delayed : bool, optional
            Whether the probes arrays are delayed (dask-awkward) or not.
            The default is True.

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"var": {"name": {"passing": passing_probes, "all": all_probes}, ...}, ...}`
            where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
            Each `"name"` is the name of eta region specified by the user.
            `passing_probes` and `all_probes` are `hist.Hist` or `hist.dask.Hist` objects.
            These are the histograms of the passing and all probes respectively.
    """
    import hist

    if delayed:
        from hist.dask import Hist
    else:
        from hist import Hist

    import egamma_tnp

    if plateau_cut is None:
        plateau_cut = 0
    if eta_regions_pt is None:
        eta_regions_pt = {
            "barrel": [0.0, 1.4442],
            "endcap": [1.566, 2.5],
        }
    if eta_regions_eta is None:
        eta_regions_eta = {"entire": [0.0, 2.5]}
    if eta_regions_phi is None:
        eta_regions_phi = {"entire": [0.0, 2.5]}

    ptbins = egamma_tnp.config.get("ptbins")
    etabins = egamma_tnp.config.get("etabins")
    phibins = egamma_tnp.config.get("phibins")

    pt_pass = passing_probes.pt
    pt_all = all_probes.pt
    eta_pass = passing_probes.eta
    eta_all = all_probes.eta
    phi_pass = passing_probes.phi
    phi_all = all_probes.phi

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass = pt_pass > plateau_cut
    plateau_mask_all = pt_all > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (
            abs(eta_pass) < region_pt[1]
        )
        eta_mask_pt_all = (abs(eta_all) > region_pt[0]) & (abs(eta_all) < region_pt[1])
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
        )
        hpt_all = Hist(
            hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
        )
        hpt_pass.fill(pt_pass[eta_mask_pt_pass])
        hpt_all.fill(pt_all[eta_mask_pt_all])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "all": hpt_all}

    for name_eta, region_eta in eta_regions_eta.items():
        eta_mask_eta_pass = (abs(eta_pass) > region_eta[0]) & (
            abs(eta_pass) < region_eta[1]
        )
        eta_mask_eta_all = (abs(eta_all) > region_eta[0]) & (
            abs(eta_all) < region_eta[1]
        )
        heta_pass = Hist(
            hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
        )
        heta_all = Hist(
            hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
        )
        heta_pass.fill(eta_pass[plateau_mask_pass & eta_mask_eta_pass])
        heta_all.fill(eta_all[plateau_mask_all & eta_mask_eta_all])

        histograms["eta"][name_eta] = {"passing": heta_pass, "all": heta_all}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (
            abs(eta_pass) < region_phi[1]
        )
        eta_mask_phi_all = (abs(eta_all) > region_phi[0]) & (
            abs(eta_all) < region_phi[1]
        )
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
        )
        hphi_all = Hist(
            hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
        )
        hphi_pass.fill(phi_pass[plateau_mask_pass & eta_mask_phi_pass])
        hphi_all.fill(phi_all[plateau_mask_all & eta_mask_phi_all])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "all": hphi_all}

    return histograms


def save_hists(path, res):
    """Save histograms to a ROOT file.

    Parameters
    ----------
        path : str
            The path to the ROOT file.
        res : dict
            A histogram dictionary of the form {"var": {"region": {"passing": hist.Hist, "all": hist.Hist}, ...}, ...}
    """
    with uproot.recreate(path) as f:
        for var, region_dict in res.items():
            for region_name, hists in region_dict.items():
                for histname, h in hists.items():
                    f[f"{var}/{region_name}/{histname}"] = h
