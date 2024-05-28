from __future__ import annotations

import pathlib

import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt

from egamma_tnp.utils import get_ratio_histogram


def _save_and_close(fig, path, close_figure):
    """Saves a figure at a given location if path is provided and optionally closes it.

    Args:
        fig (matplotlib.figure.Figure): figure to save
        path (Optional[pathlib.Path]): path where figure should be saved, or None to not
            save it
        close_figure (bool): whether to close figure after saving
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if close_figure:
        plt.close(fig)


def plot_efficiency(passing_probes, failing_or_all_probes, denominator_type="failing", **kwargs):
    """Plot the efficiency using the ratio of passing to passing + failing probes.

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
        **kwargs
            Keyword arguments to pass to hist.Hist.plot1d.

    Returns
    -------
        List[Hist1DArtists]

    """
    ratio_hist, yerr = get_ratio_histogram(passing_probes, failing_or_all_probes, denominator_type)

    return ratio_hist.plot1d(histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs)


def plot_ratio(
    passing_probes1,
    failing_or_all_probes1,
    passing_probes2,
    failing_or_all_probes2,
    label1,
    label2,
    denominator_type="failing",
    *,
    plottype="pt_low_threshold",
    figure_path=None,
    figsize=(6, 6),
    eff1_kwargs=None,
    eff2_kwargs=None,
    effratio_kwargs=None,
    cms_kwargs=None,
    legend_kwargs=None,
    efficiency_label=None,
    ratio_label=None,
):
    """Plot the ratio of two efficiencies.

    Parameters
    ----------
        passing_probes1 : hist.Hist
            The histogram of the passing probes for the first efficiency.
        failing_or_all_probes1 : hist.Hist
            The histogram of the failing or passing + failing probes for the first efficiency.
        passing_probes2 : hist.Hist
            The histogram of the passing probes for the second efficiency.
        failing_or_all_probes2 : hist.Hist
            The histogram of the failing or passing + failing probes for the second efficiency.
        label1 : str
            The label for the first efficiency.
        label2 : str
            The label for the second efficiency.
        denominator_type : str, optional
            The type of the denominator histogram.
            Can be either "failing" or "all".
            The default is "failing".
        plottype : str, optional
            The type of plot to make. Can be "pt_low_threshold", "pt_high_threshold", "eta", or "phi".
            Defaults is "pt_low_threshold".
        figure_path : str, optional
            The path where the figure should be saved, or None to not save it.
            Defaults is None.
        figsize : tuple of floats or ints, optional
            The size of the figure. Defaults is (6, 6).
        eff1_kwargs : dict, optional
            Keyword arguments to pass to hist.Hist.plot1d for the first efficiency.
        eff2_kwargs : dict, optional
            Keyword arguments to pass to hist.Hist.plot1d for the second efficiency.
        effratio_kwargs : dict, optional
            Keyword arguments to pass to matplotlib.pyplot.errorbar for the ratio.
        cms_kwargs : dict, optional
            Keyword arguments to pass to mplhep.cms.label.
        legend_kwargs : dict, optional
            Keyword arguments to pass to matplotlib.pyplot.legend.
    """
    eff1_default_kwargs = {"color": "k"}
    eff2_default_kwargs = {"color": "r"}
    effratio_default_kwargs = {
        "color": "k",
        "linestyle": "none",
        "marker": ".",
        "markersize": 10.0,
        "elinewidth": 1,
    }
    cms_default_kwargs = {
        "label": "Preliminary",
        "data": True,
        "lumi": "X",
        "year": 2023,
        "com": 13.6,
    }
    legend_default_kwargs = {}

    if eff1_kwargs is None:
        eff1_kwargs = eff1_default_kwargs
    else:
        eff1_kwargs = eff1_default_kwargs | eff1_kwargs

    if eff2_kwargs is None:
        eff2_kwargs = eff2_default_kwargs
    else:
        eff2_kwargs = eff2_default_kwargs | eff2_kwargs

    if effratio_kwargs is None:
        effratio_kwargs = effratio_default_kwargs
    else:
        effratio_kwargs = effratio_default_kwargs | effratio_kwargs

    if cms_kwargs is None:
        cms_kwargs = cms_default_kwargs
    else:
        cms_kwargs = cms_default_kwargs | cms_kwargs

    if legend_kwargs is None:
        legend_kwargs = legend_default_kwargs
    else:
        legend_kwargs = legend_default_kwargs | legend_kwargs

    if efficiency_label is None:
        efficiency_label = "Efficiency"
    if ratio_label is None:
        ratio_label = "Ratio"

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    eff1, efferr1 = get_ratio_histogram(passing_probes1, failing_or_all_probes1, denominator_type)
    eff2, efferr2 = get_ratio_histogram(passing_probes2, failing_or_all_probes2, denominator_type)
    plot_efficiency(
        passing_probes1,
        failing_or_all_probes1,
        denominator_type,
        label=label1,
        ax=ax1,
        x_axes_label=None,
        **eff1_kwargs,
    )
    plot_efficiency(
        passing_probes2,
        failing_or_all_probes2,
        denominator_type,
        label=label2,
        ax=ax1,
        x_axes_label=None,
        **eff2_kwargs,
    )
    centers = passing_probes1.axes.centers[0]
    num = eff1.values()
    denom = eff2.values()
    denom[denom == 0.0] = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / denom
        ratioyerr = np.sqrt((efferr1 / num) ** 2 + (efferr2 / denom) ** 2) * ratio
    ratioxerr = passing_probes1.axes.widths[0] / 2
    ax2.errorbar(centers, ratio, ratioyerr, ratioxerr, **effratio_kwargs)
    ax2.axhline(1, color="k", linestyle="--", linewidth=1)

    hep.cms.label(ax=ax1, **cms_kwargs)

    if plottype == "pt_low_threshold":
        ax1.set_xlim(10, 400)
        ax2.set_xlim(10, 400)
        ax2.set_xlabel(r"Offline electron $P_T$ [GeV]")
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_xticks([10, 100], [10, 100])
        ax2.set_xticks(
            [20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400],
            [20, 30, 40, 50, None, None, None, None, 200, 300, 400],
            minor=True,
        )
        legend_loc = "lower right"
    elif plottype == "pt_high_threshold":
        ax1.set_xlim(10, 400)
        ax2.set_xlim(10, 400)
        ax2.set_xlabel(r"Offline electron $P_T$ [GeV]")
        legend_loc = "lower right"
    elif plottype == "eta":
        ax1.set_xlim(-2.5, 2.5)
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_xlabel(r"Offline electron $\eta$")
        legend_loc = "lower center"
    elif plottype == "phi":
        ax1.set_xlim(-3.32, 3.32)
        ax2.set_xlim(-3.32, 3.32)
        ax2.set_xlabel(r"Offline electron $\phi$")
        legend_loc = "lower center"
    else:
        raise ValueError(f"Invalid plottype {plottype}")

    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0.7, 1.3)
    ax1.set_xlabel(None)
    ax2.set_ylabel(None)
    ax1.set_ylabel(efficiency_label)
    ax2.set_ylabel(ratio_label)
    ax1.set_xticklabels([])
    legend = ax1.legend(loc=legend_loc, **legend_kwargs)
    legend.get_title().set_multialignment("center")

    if figure_path is not None:
        figure_path = pathlib.Path(figure_path)
        _save_and_close(fig, figure_path, True)
        return fig
