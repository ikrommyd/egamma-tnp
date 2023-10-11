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


def fill_eager_histograms(res, plateau_cut, eta_regions, bins):
    """Fill eager Pt and Eta histograms of the passing and all probes.

    Parameters
    ----------
        res : tuple
            The output of Trigger.get_arrays() with compute=True.
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The histograms will be split into those eta regions.
            The default is None meaning the entire |eta| < 2.5 region is used.
        bins: dict
            The binning of the histograms.
            Should have 3 keys "ptbins", "etabins", and "phibins".
            Each key should have a list of bin edges for the Pt, Eta, and Phi histograms respectively.

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"name": [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all], ...}`
            Where each `"name"` is the name of each eta region defined by the user.
            `hpt_pass` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of the passing probes.
            `hpt_all` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of all probes.
            `heta_pass` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of the passing probes.
            `heta_all` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of all probes.
            `hphi_pass` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of the passing probes.
            `hphi_all` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of all probes.
    """
    import hist
    from hist import Hist

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
    for name, region in eta_regions.items():
        pt_mask_pass1 = pt_pass1 > plateau_cut
        pt_mask_pass2 = pt_pass2 > plateau_cut
        pt_mask_all1 = pt_all1 > plateau_cut
        pt_mask_all2 = pt_all2 > plateau_cut

        eta_mask_pass1 = (abs(eta_pass1) > region[0]) & (abs(eta_pass1) < region[1])
        eta_mask_pass2 = (abs(eta_pass2) > region[0]) & (abs(eta_pass2) < region[1])
        eta_mask_all1 = (abs(eta_all1) > region[0]) & (abs(eta_all1) < region[1])
        eta_mask_all2 = (abs(eta_all2) > region[0]) & (abs(eta_all2) < region[1])

        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name=f"hpt_{name}", label="Pt [GeV]")
        )
        hpt_all = Hist(hist.axis.Variable(ptbins, name=f"hpt_{name}", label="Pt [GeV]"))
        heta_pass = Hist(hist.axis.Variable(etabins, name=f"heta_{name}", label="eta"))
        heta_all = Hist(hist.axis.Variable(etabins, name=f"heta_{name}", label="eta"))
        hphi_pass = Hist(hist.axis.Variable(phibins, name=f"hphi_{name}", label="phi"))
        hphi_all = Hist(hist.axis.Variable(phibins, name=f"hphi_{name}", label="phi"))

        hpt_pass.fill(pt_pass1[pt_mask_pass1 & eta_mask_pass1])
        hpt_pass.fill(pt_pass2[pt_mask_pass2 & eta_mask_pass2])
        hpt_all.fill(pt_all1[pt_mask_all1 & eta_mask_all1])
        hpt_all.fill(pt_all2[pt_mask_all2 & eta_mask_all2])
        heta_pass.fill(eta_pass1[pt_mask_pass1 & eta_mask_pass1])
        heta_pass.fill(eta_pass2[pt_mask_pass2 & eta_mask_pass2])
        heta_all.fill(eta_all1[pt_mask_all1 & eta_mask_all1])
        heta_all.fill(eta_all2[pt_mask_all2 & eta_mask_all2])
        hphi_pass.fill(phi_pass1[pt_mask_pass1 & eta_mask_pass1])
        hphi_pass.fill(phi_pass2[pt_mask_pass2 & eta_mask_pass2])
        hphi_all.fill(phi_all1[pt_mask_all1 & eta_mask_all1])
        hphi_all.fill(phi_all2[pt_mask_all2 & eta_mask_all2])

        histograms[name] = [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all]

    return histograms
