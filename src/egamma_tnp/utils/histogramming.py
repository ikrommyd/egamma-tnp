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


def fill_eager_histograms(res):
    """Fill eager Pt and Eta histograms of the passing and all probes.

    Parameters
    ----------
        res : tuple
            The output of Trigger.get_arrays() with compute=True.

    Returns
    -------
        hpt_pass: hist.Hist
            The Pt histogram of the passing probes.
        hpt_all: hist.Hist
            The Pt histogram of all probes.
        heta_pass: hist.Hist
            The Eta histogram of the passing probes.
        heta_all: hist.Hist
            The Eta histogram of all probes.
        hphi_pass: hist.Hist
            The Phi histogram of the passing probes.
        hphi_all: hist.Hist
            The Phi histogram of all probes.
    """
    import json
    import os

    import hist
    from hist import Hist

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    ptbins = config["ptbins"]
    etabins = config["etabins"]
    phibins = config["phibins"]

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

    ptaxis = hist.axis.Variable(ptbins, name="pt")
    hpt_all = Hist(ptaxis)
    hpt_pass = Hist(ptaxis)

    etaaxis = hist.axis.Variable(etabins, name="eta")
    heta_all = Hist(etaaxis)
    heta_pass = Hist(etaaxis)

    phiaxis = hist.axis.Variable(phibins, name="phi")
    hphi_all = Hist(phiaxis)
    hphi_pass = Hist(phiaxis)

    hpt_pass.fill(pt_pass1)
    hpt_pass.fill(pt_pass2)
    hpt_all.fill(pt_all1)
    hpt_all.fill(pt_all2)
    heta_pass.fill(eta_pass1)
    heta_pass.fill(eta_pass2)
    heta_all.fill(eta_all1)
    heta_all.fill(eta_all2)
    hphi_pass.fill(phi_pass1)
    hphi_pass.fill(phi_pass2)
    hphi_all.fill(phi_all1)
    hphi_all.fill(phi_all2)

    return hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all
