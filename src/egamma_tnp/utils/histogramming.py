import numpy as np
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


def fill_eager_histograms(
    res,
    bins,
    plateau_cut=None,
    eta_regions_pt=None,
    eta_regions_eta=None,
    eta_regions_phi=None,
):
    """Fill eager Pt and Eta histograms of the passing and all probes.

    Parameters
    ----------
        res : tuple
            The output of Trigger.get_arrays() with compute=True for single electron triggers.
            The output of Trigger.get_arrays()["leg1"] or Trigger.get_arrays()["leg2"] with compute=True for double electron triggers
        bins: dict
            The binning of the histograms.
            Should have 3 keys "ptbins", "etabins", and "phibins".
            Each key should have a list of bin edges for the Pt, Eta, and Phi histograms respectively.
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

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"name": [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all], ...}`
            Where each `"name"` is the name of each eta region defined by the user.
            `hpt_pass` is a hist.Hist histogram of the Pt histogram of the passing probes.
            `hpt_all` is a hist.Hist histogram of the Pt histogram of all probes.
            `heta_pass` is a hist.Hist histogram of the Eta histogram of the passing probes.
            `heta_all` is a hist.Hist histogram of the Eta histogram of all probes.
            `hphi_pass` is a hist.Hist histogram of the Phi histogram of the passing probes.
            `hphi_all` is a hist.Hist histogram of the Phi histogram of all probes.
    """
    import hist
    from hist import Hist

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

    ptbins = bins["ptbins"]
    etabins = bins["etabins"]
    phibins = bins["phibins"]

    (
        pt_pass1,
        pt_pass2,
        pt_all1,
        pt_all2,
        eta_pass1,
        eta_pass2,
        eta_all1,
        eta_all2,
        phi_pass1,
        phi_pass2,
        phi_all1,
        phi_all2,
    ) = res

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass1 = pt_pass1 > plateau_cut
    plateau_mask_pass2 = pt_pass2 > plateau_cut
    plateau_mask_all1 = pt_all1 > plateau_cut
    plateau_mask_all2 = pt_all2 > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass1 = (abs(eta_pass1) > region_pt[0]) & (
            abs(eta_pass1) < region_pt[1]
        )
        eta_mask_pt_pass2 = (abs(eta_pass2) > region_pt[0]) & (
            abs(eta_pass2) < region_pt[1]
        )
        eta_mask_pt_all1 = (abs(eta_all1) > region_pt[0]) & (
            abs(eta_all1) < region_pt[1]
        )
        eta_mask_pt_all2 = (abs(eta_all2) > region_pt[0]) & (
            abs(eta_all2) < region_pt[1]
        )
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
        )
        hpt_all = Hist(
            hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
        )
        hpt_pass.fill(pt_pass1[eta_mask_pt_pass1])
        hpt_pass.fill(pt_pass2[eta_mask_pt_pass2])
        hpt_all.fill(pt_all1[eta_mask_pt_all1])
        hpt_all.fill(pt_all2[eta_mask_pt_all2])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "all": hpt_all}

    for name_eta, region_eta in eta_regions_eta.items():
        eta_mask_eta_pass1 = (abs(eta_pass1) > region_eta[0]) & (
            abs(eta_pass1) < region_eta[1]
        )
        eta_mask_eta_pass2 = (abs(eta_pass2) > region_eta[0]) & (
            abs(eta_pass2) < region_eta[1]
        )
        eta_mask_eta_all1 = (abs(eta_all1) > region_eta[0]) & (
            abs(eta_all1) < region_eta[1]
        )
        eta_mask_eta_all2 = (abs(eta_all2) > region_eta[0]) & (
            abs(eta_all2) < region_eta[1]
        )
        heta_pass = Hist(
            hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
        )
        heta_all = Hist(
            hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
        )
        heta_pass.fill(eta_pass1[plateau_mask_pass1 & eta_mask_eta_pass1])
        heta_pass.fill(eta_pass2[plateau_mask_pass2 & eta_mask_eta_pass2])
        heta_all.fill(eta_all1[plateau_mask_all1 & eta_mask_eta_all1])
        heta_all.fill(eta_all2[plateau_mask_all2 & eta_mask_eta_all2])

        histograms["eta"][name_eta] = {"passing": heta_pass, "all": heta_all}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass1 = (abs(eta_pass1) > region_phi[0]) & (
            abs(eta_pass1) < region_phi[1]
        )
        eta_mask_phi_pass2 = (abs(eta_pass2) > region_phi[0]) & (
            abs(eta_pass2) < region_phi[1]
        )
        eta_mask_phi_all1 = (abs(eta_all1) > region_phi[0]) & (
            abs(eta_all1) < region_phi[1]
        )
        eta_mask_phi_all2 = (abs(eta_all2) > region_phi[0]) & (
            abs(eta_all2) < region_phi[1]
        )
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
        )
        hphi_all = Hist(
            hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
        )
        hphi_pass.fill(phi_pass1[plateau_mask_pass1 & eta_mask_phi_pass1])
        hphi_pass.fill(phi_pass2[plateau_mask_pass2 & eta_mask_phi_pass2])
        hphi_all.fill(phi_all1[plateau_mask_all1 & eta_mask_phi_all1])
        hphi_all.fill(phi_all2[plateau_mask_all2 & eta_mask_phi_all2])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "all": hphi_all}

    return histograms
