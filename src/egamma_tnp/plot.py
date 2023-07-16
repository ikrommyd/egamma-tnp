from hist import intervals
from matplotlib import pyplot as plt

from .utils import get_ratio_histogram


def plot_efficiency(passing_probes, all_probes, **kwargs):
    """Plot the efficiency using the ratio of passing probes to all probes.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        all_probes : hist.Hist
            The histogram of all probes.
        **kwargs
            Keyword arguments to pass to hist.Hist.plot1d.

    Returns
    -------
        List[Hist1DArtists]

    """
    yerr = intervals.ratio_uncertainty(
        passing_probes.values(), all_probes.values(), uncertainty_type="efficiency"
    )
    ratio_hist = get_ratio_histogram(passing_probes, all_probes)

    return ratio_hist.plot1d(
        histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs
    )


def plot_pt_and_eta_efficiencies(
    hpt_pass, hpt_all, heta_pass, heta_all, figsize=(12, 6), kwargs_pt={}, kwargs_eta={}
):
    """Plot the Pt and Eta efficiencies.

    Parameters
    ----------
        hpt_pass : hist.Hist
            The Pt histogram of the passing probes.
        hpt_all : hist.Hist
            The Pt histogram of all probes.
        heta_pass : hist.Hist
            The Eta histogram of the passing probes.
        heta_all : hist.Hist
            The Eta histogram of all probes.
        figsize : tuple, optional
            The figure size. The default is (12, 6).
        kwargs_pt : dict, optional
            Keyword arguments to pass to hist.Hist.plot1d for the Pt histograms.
        kwargs_eta : dict, optional
            Keyword arguments to pass to hist.Hist.plot1d for the Eta histograms.

    Returns
    -------
        figue : matplotlib.figure.Figure
            The figure containing the Pt and Eta efficiencies.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = ax
    plot_efficiency(hpt_pass, hpt_all, ax=ax1, **kwargs_pt)
    plot_efficiency(heta_pass, heta_all, ax=ax2, **kwargs_eta)

    ax1.set_xlim(5, 400)
    ax1.set_xlabel("Pt [GeV]")
    ax1.set_xscale("log")

    ax2.set_xlim(-2.5, 2.5)
    ax2.set_xlabel("Eta")

    return ax1, ax2
