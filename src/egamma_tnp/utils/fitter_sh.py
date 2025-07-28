# Developed by: Sebastian Arturo Hortua, University of Kansas
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from egamma_tnp.utils.fitter_shapes import logging, mass, shape_params, sigmoid


# Setup stuff
def get_bin_info(mass):
    if mass == "Z":
        return {
            f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate([(5, 7), (7, 10), (10, 20), (20, 45), (45, 75), (75, 500)])
        }
    elif mass == "Z_muon":
        return {
            f"bin{i + 1}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(5, 7), (7, 10), (10, 20), (20, 45), (45, 75), (75, 500)])
        }
    elif mass == "JPsi":
        return {f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}") for i, (lo, hi) in enumerate([(5, 7), (7, 10), (10, 20)])}
    elif mass == "JPsi_muon":
        return {
            f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 30)])
        }


BINS_INFO = get_bin_info(mass)

FIT_CONFIGS = {}

SIGNAL_MODELS, BACKGROUND_MODELS = shape_params(mass)

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


# Logs for terminal
def print_minuit_params_table(minuit_obj, sigmoid_eff=False):
    summary_table = []
    # Header
    header = f"{'idx':>3} | {'name':^14} | {'value':^12} | {'error':^12} | {'MINOS -':^12} | {'MINOS +':^12} | {'fixed':^5} | {'lower':^10} | {'upper':^10}"
    sep = "-" * len(header)
    summary_table.append(f"\n{sep}")
    summary_table.append(header)
    summary_table.append(sep)
    # Rows
    for i, p in enumerate(minuit_obj.params):
        low = p.lower_limit if p.lower_limit is not None else ""
        high = p.upper_limit if p.upper_limit is not None else ""
        if p.is_fixed:
            err_str = "fixed"
            minos_lower = ""
            minos_upper = ""
        else:
            err_str = f"{p.error:12.6f}"
            merr = minuit_obj.merrors.get(p.name, None)
            if merr is not None:
                minos_lower = f"{merr.lower:.6f}"
                minos_upper = f"{merr.upper:.6f}"
            else:
                minos_lower = ""
                minos_upper = ""

        summary_table.append(
            f"{i:3d} | {p.name:14s} | {p.value:12.6f} | {err_str:>12s} | "
            f"{minos_lower:>12} | {minos_upper:>12} | "
            f"{p.is_fixed!s:>5s} | {low!s:10s} | {high!s:10s}"
        )

        # SYMMETRIC ERRORS
        if sigmoid_eff and p.name == "epsilon":
            raw = p.value
            sig_val = 1 / (1 + np.exp(-raw))
            deriv = sig_val * (1 - sig_val)
            fixed_flag = str(p.is_fixed)

            if p.is_fixed:
                err_col = "fixed"
            else:
                err_col = f"{(p.error * deriv):.6f}"

            # ASYMMETRIC MINOS ERRORS
            merr = minuit_obj.merrors.get("epsilon", None)
            if not p.is_fixed and merr is not None:
                lo_col = f"{(merr.lower * deriv):.6f}"
                hi_col = f"{(merr.upper * deriv):.6f}"
            else:
                lo_col = hi_col = ""

            summary_table.append(
                f"{'':>3} | {'ε_sig':14s} | {sig_val:12.6f} | {err_col:>12s} | {lo_col:>12} | {hi_col:>12} | {fixed_flag:>5s} | {'':10s} | {'':10s}"
            )

    summary_table.append(sep)
    return "\n".join(summary_table)


def print_fit_progress(m, sigmoid_eff=False):
    summary = []
    eps = m.values["epsilon"]
    if sigmoid_eff:
        eps = sigmoid(eps)
    summary.append(f"Iteration: N={m.values['N']:.1f}, ε={eps:.3f}, B_p={m.values['B_p']:.1f}, B_f={m.values['B_f']:.1f}, fval={m.fval:.1f}")
    return "\n".join(summary)


def print_fit_summary(m, popt, perr, edges_pass, args_bin, BINS_INFO, Pearson_chi2, Poisson_chi2, total_ndof, args_data, fit_type, sigmoid_eff=False):
    tol = 1e-8
    params_at_limit = []
    for p in m.params:
        if p.lower_limit is not None and abs(p.value - p.lower_limit) < tol * max(1.0, abs(p.lower_limit)):
            params_at_limit.append(p.name)
        if p.upper_limit is not None and abs(p.value - p.upper_limit) < tol * max(1.0, abs(p.upper_limit)):
            params_at_limit.append(p.name)
    any_at_limit = bool(params_at_limit)

    # 1) Get Minuit info
    fmin = m.fmin
    fcv = m.fval

    # Status flags
    valid_min = bool(m.valid)
    # EDM threshold
    if hasattr(fmin, "is_above_max_edm"):
        above_edm = fmin.is_above_max_edm
    elif hasattr(fmin, "has_above_max_edm"):
        above_edm = fmin.has_above_max_edm
    else:
        above_edm = False

    # 2) Print info on how the fit went

    # Call limit reached?
    reached_call_limit = getattr(fmin, "has_reached_call_limit", False)

    # Hesse
    hesse_failed = getattr(fmin, "hesse_failed", False)

    # Covariance accuracy
    cov_accurate = getattr(fmin, "has_accurate_covar", False)

    # 3) Build status messages
    status_map = [
        (valid_min, "Valid Minimum", "INVALID Minimum"),
        (not any_at_limit, "No parameters at limit", "Some parameters at limit"),
        (not above_edm, "Below EDM threshold", "Above EDM threshold"),
        (not reached_call_limit, "Below call limit", "Reached call limit"),
        (not hesse_failed, "Hesse ok", "Hesse failed"),
        (cov_accurate, "Covariance ok", "Covariance APPROXIMATE"),
    ]

    summary = []
    # 4) Print the combined summary box
    summary.append("\n" + "#" * 60)
    summary.append("\n" + "=" * 60)
    # Title
    summary.append(f"Fit Summary: {args_data}, {args_bin}, {fit_type}")
    summary.append("=" * 60)
    for cond, good_msg, bad_msg in status_map:
        summary.append(good_msg if cond else bad_msg)
    summary.append(f"Fit valid: {m.valid}")
    if m.covariance is None:
        summary.append("ERROR: Covariance matrix not available")
    else:
        summary.append("Covariance matrix complete")
    summary.append("-" * 60)

    # 5) Fit Type and Histogram Bin
    try:
        bin_suffix, bin_range = BINS_INFO[args_bin]
    except Exception:
        bin_suffix, bin_range = args_bin, None
    summary.append(f"Fit Type      : {args_bin}_{bin_suffix} Pass and Fail")
    if bin_range is not None:
        try:
            lo, hi = edges_pass[0], edges_pass[-1]
            summary.append(f"Histogram Bin : {lo:.1f} - {hi:.1f} GeV")
        except Exception:
            summary.append("Histogram Bin: (unknown range)")
    else:
        # fallback
        try:
            lo, hi = edges_pass[0], edges_pass[-1]
            summary.append(f"Histogram Bin: \t{lo:.1f} - {hi:.1f} GeV")
        except Exception:
            pass
    summary.append(f"Fit Success   : {'Yes' if m.valid else 'No'}")
    summary.append("=" * 60)

    # 6) Fit Parameters table, can add more if needed
    summary.append("Fit Parameters:")
    if "N" in popt:
        summary.append(f"  Total N          = {popt['N']:.1f} ± {perr.get('N', float('nan')):.1f}")
    if "epsilon" in popt:
        if sigmoid_eff:
            eff = sigmoid(popt["epsilon"])
            eff_err = abs(perr.get("epsilon", float("nan")) * eff * (1 - eff))
            summary.append(f"  Efficiency ε     = {eff:.6f} ± {eff_err:.10f} (sigmoid)")
        else:
            summary.append(f"  Efficiency ε     = {popt['epsilon']:.6f} ± {perr.get('epsilon', float('nan')):.10f}")
    if "B_p" in popt:
        summary.append(f"  Background B_p   = {popt['B_p']:.1f} ± {perr.get('B_p', float('nan')):.1f}")
    if "B_f" in popt:
        summary.append(f"  Background B_f   = {popt['B_f']:.1f} ± {perr.get('B_f', float('nan')):.1f}")
    summary.append("-" * 60)

    # 7) Goodness-of-Fit
    summary.append("Goodness-of-Fit:")
    chi2 = fcv
    reduced_chi2 = getattr(m.fmin, "reduced_chi2", None)
    summary.append(f"  NLL     χ² / ndf = {chi2:.2f} / {total_ndof} = {reduced_chi2:.3f}")
    # Print Pearson and Poisson χ² / ndf
    Pearson_red = Pearson_chi2 / total_ndof if total_ndof else None
    Poisson_red = Poisson_chi2 / total_ndof if total_ndof else None
    summary.append(f"  Pearson χ² / ndf = {Pearson_chi2:.2f} / {total_ndof} = {Pearson_red:.3f}")
    summary.append(f"  Poisson χ² / ndf = {Poisson_chi2:.2f} / {total_ndof} = {Poisson_red:.3f}")
    summary.append("-" * 60)

    # 8) Efficiency line
    if "epsilon" in popt:
        try:
            lo, hi = edges_pass[0], edges_pass[-1]
            if sigmoid_eff:
                eff = sigmoid(popt["epsilon"])
                eff_err = abs(perr.get("epsilon", float("nan")) * eff * (1 - eff))
                summary.append("Efficiency:")
                summary.append(f"  ε = {eff:.4f} ± {eff_err:.4f} (bin {lo:.1f} - {hi:.1f} GeV, sigmoid)")
            else:
                summary.append("Efficiency:")
                summary.append(f"  ε = {popt['epsilon']:.4f} ± {perr.get('epsilon', float('nan')):.4f} (bin {lo:.1f} - {hi:.1f} GeV)")
        except Exception:
            if sigmoid_eff:
                eff = sigmoid(popt["epsilon"])
                eff_err = abs(perr.get("epsilon", float("nan")) * eff * (1 - eff))
                summary.append(f"Efficiency: ε = {eff:.4f} ± {eff_err:.4f} (sigmoid)")
            else:
                summary.append(f"Efficiency: ε = {popt['epsilon']:.4f} ± {perr.get('epsilon', float('nan')):.4f}")
    # Bottom border
    summary.append("=" * 60 + "\n")
    summary.append("#" * 60)

    return "\n".join(summary)


# Things to make simultaneous model
class PassFailPlotter:
    def __init__(self, cost_func_pass, error_func_pass, cost_func_fail, error_func_fail, n_bins_pass, edges_pass, edges_fail, fit_type, sigmoid_eff=False):
        self.cost = cost_func_pass
        self.error_pass = error_func_pass
        self.cost_fail = cost_func_fail
        self.error_fail = error_func_fail
        self.n_bins_pass = n_bins_pass
        self.edges_pass = edges_pass
        self.edges_fail = edges_fail
        self.param_names = FIT_CONFIGS[fit_type]["param_names"]
        self.signal_func = FIT_CONFIGS[fit_type]["signal_pdf"]
        self.bg_func = FIT_CONFIGS[fit_type]["background_pdf"]
        self.signal_param_names = SIGNAL_MODELS[fit_type.split("_")[0]]["params"]
        self.bg_param_names = BACKGROUND_MODELS[fit_type.split("_")[1]]["params"]
        self.sigmoid_eff = sigmoid_eff
        plt.rcParams.update({"font.size": 8})  # sets global font size

    def __call__(self, args):
        param_dict = dict(zip(self.param_names, args))

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


def create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=False, sigmoid_eff=False):
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
    params_dict = dict(zip(param_names, params))

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
            logging.info(f"Histogram: {hist_name}")
            return {"values": values, "edges": edges, "errors": obj.errors(), "is_mc": is_mc}
    return None


# Plot fits
def plot_combined_fit(results, plot_dir=".", data_type="DATA", fixed_params=None, sigmoid_eff=False, args_abseta=None, args_mass=None):
    if results is None:
        logging.warning("No results to plot")
        return None, None  # Return None if no results

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
