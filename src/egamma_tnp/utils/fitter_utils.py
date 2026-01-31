from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from iminuit import Minuit, cost

from egamma_tnp.utils.fitter_models import shape_params, sigmoid
from egamma_tnp.utils.logger_utils import print_fit_summary_rich

logger = logging.getLogger(__name__)


# Setup stuff
def get_bin_info(mass, bin_ranges):
    if mass == "Z":
        return {f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate(bin_ranges)}
    elif mass == "Z_muon":
        return {f"bin{i + 1}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate(bin_ranges)}
    elif mass == "JPsi":
        return {f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate(bin_ranges)}
    elif mass == "JPsi_muon":
        return {f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate(bin_ranges)}


def setup(config):
    mass = config["mass"]
    bin_ranges = config["fit"].get("bin_ranges", [])
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)

    BINS_INFO = get_bin_info(mass, bin_ranges)
    FIT_CONFIGS = {}

    SIGNAL_MODELS, BACKGROUND_MODELS = shape_params(mass, x_min, x_max)

    for sig_name, sig_config in SIGNAL_MODELS.items():
        for bg_name, bg_config in BACKGROUND_MODELS.items():
            fit_type = f"{sig_name}_{bg_name}"

            # Build parameter names list
            param_names = ["N", "epsilon", "B_p", "B_f"]

            # Add SHARED signal parameters (not pass/fail)
            param_names.extend(sig_config["params"])

            # Add pass/fail versions of background parameters
            for p in bg_config["params"]:
                param_names.extend([f"{p}_pass", f"{p}_fail"])

            # Build bounds dictionary
            bounds = {"N": (0, 100000, np.inf), "epsilon": (0, 0.9, 1), "B_p": (0, 10000, np.inf), "B_f": (0, 10000, np.inf)}

            # Signal Bounds
            for p, b in sig_config["bounds"].items():
                bounds[p] = b

            # Background Bounds
            for p, b in bg_config["bounds"].items():
                bounds[f"{p}_pass"] = b
                bounds[f"{p}_fail"] = b

            FIT_CONFIGS[fit_type] = {
                "param_names": param_names,
                "bounds": bounds,
                "signal_pdf": sig_config["pdf"],
                "signal_cdf": sig_config.get("cdf"),
                "background_pdf": bg_config["pdf"],
                "background_cdf": bg_config.get("cdf"),
            }
    return BINS_INFO, SIGNAL_MODELS, BACKGROUND_MODELS, FIT_CONFIGS


# Things to make simultaneous model
class PassFailPlotter:
    def __init__(
        self, fitter_config, cost_func_pass, error_func_pass, cost_func_fail, error_func_fail, n_bins_pass, edges_pass, edges_fail, fit_type, sigmoid_eff=False
    ):
        self.cost = cost_func_pass
        self.error_pass = error_func_pass
        self.cost_fail = cost_func_fail
        self.error_fail = error_func_fail
        self.n_bins_pass = n_bins_pass
        self.edges_pass = edges_pass
        self.edges_fail = edges_fail
        BINS_INFO, SIGNAL_MODELS, BACKGROUND_MODELS, FIT_CONFIGS = setup(fitter_config)
        self.param_names = FIT_CONFIGS[fit_type]["param_names"]
        self.signal_func = FIT_CONFIGS[fit_type]["signal_pdf"]
        self.bg_func = FIT_CONFIGS[fit_type]["background_pdf"]
        self.signal_param_names = SIGNAL_MODELS[fit_type.split("_")[0]]["params"]
        self.bg_param_names = BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]
        self.sigmoid_eff = sigmoid_eff
        plt.rcParams.update({"font.size": 8})  # sets global font size

    def __call__(self, args):
        param_dict = dict(zip(self.param_names, args, strict=True))

        # Split the data
        data_pass = self.cost.data
        data_fail = self.cost_fail.data
        data_pass = data_pass[: self.n_bins_pass]
        data_fail = data_fail[: self.n_bins_pass]

        cx_pass = 0.5 * (self.edges_pass[:-1] + self.edges_pass[1:])
        cx_fail = 0.5 * (self.edges_fail[:-1] + self.edges_fail[1:])

        widths_pass = np.diff(self.edges_pass)
        widths_fail = np.diff(self.edges_fail)

        # Rebuild signal and background
        signal_params = [param_dict[p] for p in self.signal_param_names]
        bg_pass_params = [param_dict[f"{p}_pass"] for p in self.bg_param_names]
        bg_fail_params = [param_dict[f"{p}_fail"] for p in self.bg_param_names]

        signal_pass = self.signal_func(cx_pass, *signal_params)
        signal_fail = self.signal_func(cx_fail, *signal_params)

        bg_pass = self.bg_func(cx_pass, *bg_pass_params)
        bg_fail = self.bg_func(cx_fail, *bg_fail_params)

        N = param_dict["N"]
        if self.sigmoid_eff:
            epsilon = sigmoid(param_dict["epsilon"])
        else:
            epsilon = param_dict["epsilon"]
        B_p = param_dict["B_p"]
        B_f = param_dict["B_f"]

        if N <= (B_p + B_f):
            # Don't plot if constraint is violated
            return

        signal_y_pass = (N - (B_p + B_f)) * epsilon * signal_pass
        signal_y_fail = (N - (B_p + B_f)) * (1 - epsilon) * signal_fail
        bg_y_pass = B_p * bg_pass
        bg_y_fail = B_f * bg_fail
        total_pass = signal_y_pass + bg_y_pass

        # model yields → densities
        signal_y_pass = signal_y_pass * widths_pass
        bg_y_pass = bg_y_pass * widths_pass
        total_pass = total_pass * widths_pass

        total_fail = signal_y_fail + bg_y_fail

        signal_y_fail = signal_y_fail * widths_fail
        bg_y_fail = bg_y_fail * widths_fail
        total_fail = total_fail * widths_fail

        # Plot pass
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("Pass")
        plt.errorbar(cx_pass, data_pass, yerr=self.error_pass, fmt="o", color="black", label="Data")
        plt.stairs(bg_y_pass, self.edges_pass, fill=True, color="orange", label="Background")
        plt.stairs(total_pass, self.edges_pass, baseline=bg_y_pass, fill=True, color="skyblue", label="Signal")
        plt.stairs(total_pass, self.edges_pass, color="navy", label="Total Fit")
        plt.legend()

        # Plot fail
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Fail")
        plt.errorbar(cx_fail, data_fail, yerr=self.error_fail, fmt="o", color="black", label="Data")
        plt.stairs(bg_y_fail, self.edges_fail, fill=True, color="orange", label="Background")
        plt.stairs(total_fail, self.edges_fail, baseline=bg_y_fail, fill=True, color="skyblue", label="Signal")
        plt.stairs(total_fail, self.edges_fail, color="navy", label="Total Fit")
        plt.legend()

        plt.tight_layout()


class CombinedCost:
    def __init__(self, cost1, cost2):
        self.cost1 = cost1
        self.cost2 = cost2
        self.ndata = cost1.ndata + cost2.ndata

    def __call__(self, *params):
        return self.cost1(*params) + self.cost2(*params)


def create_fixed_param_wrapper(func, fixed_params):
    def wrapped(x, *free_params):
        full_params = []
        free_idx = 0
        for i in range(len(fixed_params) + len(free_params)):
            if i in fixed_params:
                full_params.append(fixed_params[i])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1
        return func(x, *full_params)

    return wrapped


def calculate_custom_chi2(values, errors, model, n_params):
    mask = (errors > 0) & (model > 0) & (values >= 0)

    # Calculate Pearson chi2
    Pearson_chi2 = np.sum(((values[mask] - model[mask]) / errors[mask]) ** 2)

    # Calculate Poisson chi2
    safe_ratio = np.zeros_like(values)
    ratio = np.divide(values, model, out=safe_ratio, where=(values > 0) & (model > 0))
    log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=(ratio > 0))
    Poisson_terms = model - values + values * log_ratio
    valid_terms = np.isfinite(Poisson_terms) & (Poisson_terms >= 0)
    Poisson_chi2 = 2 * np.sum(Poisson_terms[valid_terms])

    # Calculate degrees of freedom
    ndof = mask.sum() - n_params

    return Pearson_chi2, Poisson_chi2, ndof


def create_combined_model(fitter_config, fit_type, edges_pass, edges_fail, *params, use_cdf=False, sigmoid_eff=False):
    BINS_INFO, SIGNAL_MODELS, BACKGROUND_MODELS, FIT_CONFIGS = setup(fitter_config)
    config = FIT_CONFIGS[fit_type]

    # If either CDF is missing, fall back to PDF mode for the entire region
    if use_cdf and (config["signal_cdf"] is None or config["background_cdf"] is None):
        use_cdf = False

    signal_func = config["signal_cdf"] if use_cdf else config["signal_pdf"]
    bg_func = config["background_cdf"] if use_cdf else config["background_pdf"]
    param_names = config["param_names"]

    # For CDF mode, edges are used; for PDF mode, bin centers are used
    if use_cdf:
        x = edges_pass
        y = edges_fail
    else:
        x = 0.5 * (edges_pass[:-1] + edges_pass[1:])
        y = 0.5 * (edges_fail[:-1] + edges_fail[1:])
    params_dict = dict(zip(param_names, params, strict=True))

    # Shared signal parameters
    signal_params = [params_dict[p] for p in SIGNAL_MODELS[fit_type.split("_")[0]]["params"]]

    # Background parameters
    bg_pass_params = [params_dict[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]]
    bg_fail_params = [params_dict[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]]

    signal_pass = signal_func(x, *signal_params)
    bg_pass = bg_func(x, *bg_pass_params)
    signal_fail = signal_func(y, *signal_params)
    bg_fail = bg_func(y, *bg_fail_params)

    N = params_dict["N"]
    if sigmoid_eff:
        epsilon = sigmoid(params_dict["epsilon"])
    else:
        epsilon = params_dict["epsilon"]
    B_p = params_dict["B_p"]
    B_f = params_dict["B_f"]

    # Avoid regions with N <= (B_p + B_f) by returning large values if N is too small
    if N <= (B_p + B_f):
        large_value = 1e10
        return np.full_like(x, large_value), np.full_like(y, large_value)

    # FULL MODEL --> (N - (B_p + B_f)) * ( epsilon * signal_pass + (1 - epsilon) * signal_fail) + B_p * bg_pass + B_f * bg_fail

    result_pass = (N - (B_p + B_f)) * epsilon * signal_pass + B_p * bg_pass
    result_fail = (N - (B_p + B_f)) * (1 - epsilon) * signal_fail + B_f * bg_fail

    result_pass = np.clip(result_pass, 1e-10, None)
    result_fail = np.clip(result_fail, 1e-10, None)
    return result_pass, result_fail


# Load histograms
def load_histogram(root_file, hist_name, data_label):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            is_mc = ("MC" in data_label) or ("MC" in hist_name)
            # logger.info(f"Histogram: {hist_name}")
            return {"values": values, "edges": edges, "errors": obj.errors(), "is_mc": is_mc}
    return None


def fig_to_array(fig):
    """Convert a Matplotlib figure to a NumPy array (RGBA)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    # Convert ARGB to RGBA
    buf = buf[:, :, [1, 2, 3, 0]]
    return buf


# Plot fits
def plot_combined_fit(fitter_config, results, plot_dir=".", data_type="DATA", fixed_params=None, sigmoid_eff=False, args_abseta=None, args_mass=None):
    if results is None:
        logger.warning("No results to plot")
        return None, None  # Return None if no results

    BINS_INFO, SIGNAL_MODELS, BACKGROUND_MODELS, FIT_CONFIGS = setup(fitter_config)
    fixed_params = fixed_params or {}
    fit_type = results["type"]
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_pdf"]
    bg_func = config["background_pdf"]
    params = results["popt"]
    perr = results["perr"]

    # Get signal and background model names for the legend
    signal_model_name = {"dcb": "Double Crystal Ball", "dv": "Double Voigtian", "g": "Gaussian", "cb_g": "Crystal Ball + Gaussian"}.get(
        fit_type.split("_")[0], "Unknown Signal"
    )

    background_model_name = {
        "ps": "Phase Space",
        "lin": "Linear",
        "exp": "Exponential",
        "cheb": "Chebyshev Polynomial",
        "cms": "CMS Shape",
        "bpoly": "Bernstein Polynomial",
    }.get(fit_type.split("_")[1], "Unknown Background")

    x = np.linspace(results["x_min"], results["x_max"], 1000)
    signal_params = [params[p] for p in SIGNAL_MODELS[fit_type.split("_")[0]]["params"]]

    def format_param(name, value, error, fixed_params):
        if name in fixed_params:
            return f"{name} = {fixed_params[name]:.3f} (fixed)"
        elif np.isnan(value):
            return f"{name} = NaN"
        elif np.isinf(value):
            return f"{name} = Infinity"
        elif error == 0:
            return f"{name} = {value:.3f} (fixed)"
        else:
            return f"{name} = {value:.3f} ± {error:.6f}"

    # Compute efficiency and error for display
    if sigmoid_eff:
        eff = sigmoid(params["epsilon"])
        eff_err = abs(perr["epsilon"] * eff * (1 - eff))
        eff_label = " (sigmoid)"
    else:
        eff = params["epsilon"]
        eff_err = perr["epsilon"]
        eff_label = ""

    # --- PASS plot ---
    fig_pass, ax_pass = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")

    bin_widths_pass = results["bin_widths_pass"]
    bin_widths_fail = results["bin_widths_fail"]

    bg_pass_params = [params[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]]
    signal_pass = (params["N"] - (params["B_p"] + params["B_f"])) * eff * signal_func(x, *signal_params)
    bg_pass = params["B_p"] * bg_func(x, *bg_pass_params)
    total_pass = signal_pass + bg_pass

    signal_pass = signal_pass * np.mean(bin_widths_pass)
    bg_pass = bg_pass * np.mean(bin_widths_pass)
    total_pass = total_pass * np.mean(bin_widths_pass)

    ax_pass.errorbar(
        results["centers_pass"], results["values_pass"], yerr=results["errors_pass"], fmt="o", markersize=6, capsize=3, color="royalblue", label="Data (Pass)"
    )
    ax_pass.plot(x, total_pass, "k-", label="Total fit")
    ax_pass.plot(x, signal_pass, "g--", label=f"Signal ({signal_model_name})")
    ax_pass.plot(x, bg_pass, "r--", label=f"Background ({background_model_name})")

    ax_pass.set_xlabel("$m_{ee}$ [GeV]", fontsize=30)
    ax_pass.set_ylabel("Events / GeV", fontsize=30)
    bin_suffix, bin_range = BINS_INFO[results["bin"]]
    ax_pass.set_title(f"{data_type.replace('_', ' ')}: {bin_range} GeV (Pass)", pad=10)

    chi2_red = results["reduced_chi_squared"]
    signal_params_text = "\n".join([format_param(p, params[p], results["perr"][p], fixed_params) for p in SIGNAL_MODELS[fit_type.split("_")[0]]["params"]])
    bg_params_text = "\n".join(
        [
            format_param(f"{p}_pass", params[f"{p}_pass"], results["perr"][f"{p}_pass"], fixed_params)
            for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]
        ]
    )

    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {eff:.6f} ± {eff_err:.6f}{eff_label}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        "",
        f"Converged: {results['converged']}",
        f"\nSignal yield: {params['N'] * eff:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"NLL: χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
        f"Pearson: χ²/ndof = {results['Pearson_chi2']:.1f}/{results['total_ndof']} = {results['Pearson_tot_red_chi2']:.2f}",
        f"Poisson: χ²/ndof = {results['Poisson_chi2']:.1f}/{results['total_ndof']} = {results['Poisson_tot_red_chi2']:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text,
    ]
    ax_pass.legend(loc="upper right", fontsize=10)
    ax_pass.text(
        0.02,
        0.98,
        "\n".join(info_text),
        transform=ax_pass.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
    )

    os.makedirs(plot_dir, exist_ok=True)

    # --- FAIL plot ---
    fig_fail, ax_fail = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")

    bg_fail_params = [params[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]]
    signal_fail = (params["N"] - (params["B_p"] + params["B_f"])) * (1 - eff) * signal_func(x, *signal_params)
    bg_fail = params["B_f"] * bg_func(x, *bg_fail_params)
    total_fail = signal_fail + bg_fail

    signal_fail = signal_fail * np.mean(bin_widths_fail)
    bg_fail = bg_fail * np.mean(bin_widths_fail)
    total_fail = total_fail * np.mean(bin_widths_fail)

    ax_fail.errorbar(
        results["centers_fail"], results["values_fail"], yerr=results["errors_fail"], fmt="o", markersize=6, capsize=3, color="royalblue", label="Data (Fail)"
    )
    ax_fail.plot(x, total_fail, "k-", label="Total fit")
    ax_fail.plot(x, signal_fail, "purple", linestyle="--", label=f"Signal ({signal_model_name})")
    ax_fail.plot(x, bg_fail, "r--", label=f"Background ({background_model_name})")

    ax_fail.set_xlabel("$m_{ee}$ [GeV]", fontsize=30)
    ax_fail.set_ylabel("Events / GeV", fontsize=30)
    bin_suffix, bin_range = BINS_INFO[results["bin"]]
    ax_fail.set_title(f"{data_type.replace('_', ' ')}: {bin_range} GeV (Fail)", pad=10)
    bg_params_text_fail = "\n".join(
        [
            format_param(f"{p}_fail", params[f"{p}_fail"], results["perr"][f"{p}_fail"], fixed_params)
            for p in BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]
        ]
    )
    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {eff:.6f} ± {eff_err:.6f}{eff_label}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        "",
        f"Converged: {results['converged']}",
        f"\nSignal yield: {params['N'] * (1 - eff):.1f}",
        f"Bkg yield: {params['B_f']:.1f}",
        f"NLL: χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
        f"Pearson: χ²/ndof = {results['Pearson_chi2']:.1f}/{results['total_ndof']} = {results['Pearson_tot_red_chi2']:.2f}",
        f"Poisson: χ²/ndof = {results['Poisson_chi2']:.1f}/{results['total_ndof']} = {results['Poisson_tot_red_chi2']:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text_fail,
    ]
    ax_fail.legend(loc="upper right", fontsize=10)
    ax_fail.text(
        0.02,
        0.98,
        "\n".join(info_text),
        transform=ax_fail.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
    )

    return fig_pass, fig_fail


fit_prog = []
fit_summary = []
fit_text_sum = []


def fit_function(
    fitter_config,
    fit_type,
    hist_pass,
    hist_fail,
    x_min,
    x_max,
    fixed_params=None,
    use_cdf=False,
    interactive=False,
    args_bin=None,
    args_data=None,
    sigmoid_eff=False,
    args_mass=None,
    data_name=None,
    mc_name=None,
):
    BINS_INFO, SIGNAL_MODELS, BACKGROUND_MODELS, FIT_CONFIGS = setup(fitter_config)
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]

    if use_cdf and (config["signal_cdf"] is None or config["background_cdf"] is None):
        logger.warning(f"[Warning] Model '{fit_type}' missing CDF(s). Disabling CDF mode.")
        use_cdf = False
    param_names = config["param_names"]

    # Get data for pass
    centers_pass = (hist_pass["edges"][:-1] + hist_pass["edges"][1:]) / 2
    edges_pass = hist_pass["edges"]
    values_pass = hist_pass["values"]
    errors_pass = hist_pass["errors"]

    # Get data for fail
    centers_fail = (hist_fail["edges"][:-1] + hist_fail["edges"][1:]) / 2
    edges_fail = hist_fail["edges"]
    values_fail = hist_fail["values"]
    errors_fail = hist_fail["errors"]

    # Crop region to the x range
    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)

    edge_mask = np.zeros(len(edges_pass), dtype=bool)
    edge_mask[:-1] |= mask_pass
    edge_mask[1:] |= mask_pass

    centers_pass = centers_pass[mask_pass]
    edges_pass = edges_pass[edge_mask]
    values_pass = values_pass[mask_pass]
    errors_pass = errors_pass[mask_pass]

    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)
    edge_mask = np.zeros(len(edges_fail), dtype=bool)
    edge_mask[:-1] |= mask_fail
    edge_mask[1:] |= mask_fail

    centers_fail = centers_fail[mask_fail]
    edges_fail = edges_fail[edge_mask]
    values_fail = values_fail[mask_fail]
    errors_fail = errors_fail[mask_fail]

    if args_mass == "Z" or args_mass == "Z_muon":
        # Calculate data-based initial guesses
        N_p0 = np.sum(values_pass) + np.sum(values_fail)
        B_p_p0 = max(1, np.median(values_pass[-10:]) * len(values_pass))
        B_f_p0 = max(1, np.median(values_fail[-10:]) * len(values_fail))

        # Scale fixed parameters if present
        for name in ["N", "epsilon", "B_p", "B_f"]:
            if name in fixed_params:
                fixed_params[name]

        # Update bounds with data-based values
        bounds = config["bounds"].copy()
        bounds.update(
            {
                "N": (B_p_p0 + B_f_p0, N_p0 * 10, np.inf),
                "B_p": (0, B_p_p0 / 4, np.inf),
                "B_f": (0, B_f_p0, np.inf),
            }
        )

    elif args_mass == "JPsi" or args_mass == "JPsi_muon":
        # Calculate data-based initial guesses
        N_p0 = (np.sum(values_pass) + np.sum(values_fail)) * 0.1
        sideband_mask = (centers_pass < 2.8) | (centers_pass > 3.4)
        sideband_mask_fail = (centers_fail < 2.8) | (centers_fail > 3.4)
        sideband_values_pass = values_pass[sideband_mask]
        sideband_values_fail = values_fail[sideband_mask_fail]

        if len(sideband_values_pass) == 0:
            B_p_p0 = 1  # fallback guess
        else:
            B_p_p0 = max(1, np.median(sideband_values_pass) * len(values_pass))

        if len(sideband_values_fail) == 0:
            B_f_p0 = 1  # fallback guess
        else:
            B_f_p0 = max(1, np.median(sideband_values_fail) * len(values_fail))

        # Scale fixed parameters if present
        for name in ["N", "epsilon", "B_p", "B_f"]:
            if name in fixed_params:
                fixed_params[name]

        # Update bounds with data-based values
        bounds = config["bounds"].copy()
        bounds.update(
            {
                "N": (0, N_p0 * 10, np.inf),
                "B_p": (0, B_p_p0, np.inf),
                "B_f": (0, B_f_p0, np.inf),
            }
        )

    # Set epsilon bounds and initial guess depending on sigmoid
    if sigmoid_eff:
        bounds["epsilon"] = (-10, 0, 10)  # raw parameter, maps to (0.002, 0.998)
    else:
        bounds["epsilon"] = (0, 0.95, 1)

    # Prepare initial parameter guesses
    p0 = []
    bounds_low = []
    bounds_high = []
    initial_guesses = {}  # All initial guesses are stored here

    for name in param_names:
        if name in fixed_params:
            initial_guesses[name] = fixed_params[name]
            continue
        else:
            b = bounds[name]
            if name == "epsilon" and sigmoid_eff:
                # Set initial guess for raw parameter so sigmoid(epsilon) ~ 0.9
                initial_guesses[name] = 2.2  # sigmoid(2.2) ≈ 0.9
            else:
                initial_guesses[name] = b[1]
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    def model_approx_pass(edges, *params):
        result_pass, _ = create_combined_model(fitter_config, fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_approx_fail(edges, *params):
        _, result_fail = create_combined_model(fitter_config, fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_fail

    def model_cdf_pass(edges, *params):
        result_pass, _ = create_combined_model(fitter_config, fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_cdf_fail(edges, *params):
        _, result_fail = create_combined_model(fitter_config, fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_fail

    bin_widths_pass = np.diff(edges_pass)
    bin_widths_fail = np.diff(edges_fail)

    if use_cdf:
        model_pass = model_cdf_pass
        model_fail = model_cdf_fail
    else:
        model_pass = model_approx_pass
        model_fail = model_approx_fail

    # Cost functions depending on if using CDF or PDF
    if use_cdf:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass)
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail)
        c_fail.errdef = Minuit.LIKELIHOOD
    else:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass, use_pdf="approximate")
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail, use_pdf="approximate")
        c_fail.errdef = Minuit.LIKELIHOOD

    # Create CombinedCost object
    c = CombinedCost(c_pass, c_fail)
    # Set error definition to NLL
    c.errdef = Minuit.LIKELIHOOD

    init = initial_guesses

    # Create Minuit object
    m = Minuit(c, *[init[name] for name in param_names], name=param_names)

    for name in param_names:
        if name in fixed_params:
            init[name] = fixed_params[name]
            m.fixed[name] = True
        elif name in bounds:
            m.limits[name] = (bounds[name][0], bounds[name][2])

    m.hesse()

    # Interactive Fitter
    plotter = PassFailPlotter(
        fitter_config, c_pass, errors_pass, c_fail, errors_fail, len(edges_pass), edges_pass, edges_fail, fit_type, sigmoid_eff=sigmoid_eff
    )

    if interactive:
        m.strategy = 1
        m.interactive(plotter)
    else:
        # Set default strategy
        m.strategy = 1

        # Try a full Migrad first
        m.migrad(iterate=50)
        m.hesse()

        if not m.valid:
            fall_back_params = []

            m.strategy = 2
            m.migrad(iterate=1)
            # First fix all parameters
            for p in m.parameters:
                m.fixed[p] = True
            for p in m.parameters:
                m.fixed[p] = False
                m.strategy = 2
                m.migrad()
                if not m.fmin.is_valid:
                    fall_back_params.append(p)
                    m.fixed[p] = True
            m.fixed[p] = True
            for p in fall_back_params:
                m.fixed[p] = True

            # Final Migrad + Hesse
            m.strategy = 1
            m.migrad(iterate=50)
            m.hesse()

    plt.close("all")

    for param in m.parameters:
        try:
            m.minos(param)
        except Exception as e:
            logger.debug(f"MINOS failed: {e!s}")

    # Print results
    m.print_level = 0

    # 2. Extract results
    fcv = m.fval
    popt = m.values.to_dict()
    perr = m.errors.to_dict()

    # NLL chi2and ndof
    chi2 = fcv
    dof = m.ndof
    reduced_chi2 = chi2 / dof

    # compute model predictions at bin centers or edges as appropriate
    model_pass_vals = model_approx_pass(edges_pass, *m.values)
    model_fail_vals = model_approx_fail(edges_fail, *m.values)

    # Scale model predictions by bin widths
    model_pass_vals = model_pass_vals * bin_widths_pass
    model_fail_vals = model_fail_vals * bin_widths_fail

    # Calculate Pearson and Poisson Chi2 and ndof
    n_params = len([name for name in m.parameters if not m.fixed[name]])
    Pearson_chi2_pass, Poisson_chi2_pass, ndof_pass = calculate_custom_chi2(values_pass, errors_pass, model_pass_vals, n_params)
    Pearson_chi2_fail, Poisson_chi2_fail, ndof_fail = calculate_custom_chi2(values_fail, errors_fail, model_fail_vals, n_params)
    Pearson_chi2 = Pearson_chi2_pass + Pearson_chi2_fail
    Poisson_chi2 = Poisson_chi2_pass + Poisson_chi2_fail
    total_ndof = (len(values_pass) + len(values_fail)) - n_params
    Pearson_tot_red_chi2 = Pearson_chi2 / total_ndof
    Poisson_tot_red_chi2 = Poisson_chi2 / total_ndof

    # Output fit summary to terminal
    summary_text = print_fit_summary_rich(
        m,
        popt,
        perr,
        edges_pass,
        args_bin,
        BINS_INFO,
        Pearson_chi2,
        Poisson_chi2,
        total_ndof,
        args_data,
        fit_type,
        sigmoid_eff=sigmoid_eff,
        DATA_NAME=data_name,
        MC_NAME=mc_name,
    )

    fit_text_sum.append(summary_text)

    # Check convergence
    if m.fmin.is_valid:
        converged = "Fit converged: minimum is valid."
    else:
        issues = []
        if m.fmin.has_parameters_at_limit:
            issues.append("\nparameters at limit")
        if m.fmin.edm > m.tol:
            issues.append(f"\nEDM too high (EDM = {m.fmin.edm:.2g}, tol = {m.tol:.2g})")
        if not m.fmin.has_valid_parameters:
            issues.append("\ninvalid Hesse (covariance matrix)")
        if m.fmin.has_reached_call_limit:
            issues.append(f"\ncall limit reached (Nfcn = {m.fmin.nfcn})")

        if not issues:
            issues.append("unknown issue")

        converged = "Fit did not converge: " + "; ".join(issues)

    minos_errors = {}
    for param in m.parameters:
        if param in m.merrors:
            minos_errors[param] = {"lower": m.merrors[param].lower, "upper": m.merrors[param].upper}
        else:
            minos_errors[param] = {
                "lower": -m.errors[param],  # Fall back to symmetric errors
                "upper": m.errors[param],
            }

    # Return results
    results = {
        "minos_errors": minos_errors,
        "m": m,
        "type": fit_type,
        "popt": popt,
        "perr": perr,
        "cov": m.covariance,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "dof": dof,
        "Pearson_chi2": Pearson_chi2,
        "Poisson_chi2": Poisson_chi2,
        "Pearson_tot_red_chi2": Pearson_tot_red_chi2,
        "Poisson_tot_red_chi2": Poisson_tot_red_chi2,
        "total_ndof": total_ndof,
        "success": m.valid,
        "message": m.fmin,
        "param_names": param_names,
        "centers_pass": centers_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
        "values_fail": values_fail,
        "errors_fail": errors_fail,
        "x_min": x_min,
        "x_max": x_max,
        "converged": converged,
        "bin_widths_pass": bin_widths_pass,
        "bin_widths_fail": bin_widths_fail,
        "sum_text": summary_text,
    }

    results["bin"] = args_bin
    return results
