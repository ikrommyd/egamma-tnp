# Developed by: Sebastian Arturo Hortua, University of Kansas

import argparse
import os
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import voigt_profile
from scipy.stats import norm
from scipy.interpolate import BPoly
from iminuit import cost, Minuit
from scipy import special
from numba_stats import cmsshape
import os
import json

from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to JSON config file")
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)
    mass = config["mass"]
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)

def print_minuit_params_table(minuit_obj, sigmoid_eff=False):
    # Header
    header = f"{'idx':>3} | {'name':^14} | {'value':^12} | {'error':^12} | {'MINOS -':^12} | {'MINOS +':^12} | {'fixed':^5} | {'lower':^10} | {'upper':^10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    # Rows
    for i, p in enumerate(minuit_obj.params):
        low  = p.lower_limit if p.lower_limit is not None else ""
        high = p.upper_limit if p.upper_limit is not None else ""
        if p.is_fixed:
            err_str     = "fixed"
            minos_lower = ""
            minos_upper = ""
        else:
            err_str     = f"{p.error:12.6f}"
            merr        = minuit_obj.merrors.get(p.name, None)
            if merr is not None:
                minos_lower = f"{merr.lower:.6f}"
                minos_upper = f"{merr.upper:.6f}"
            else:
                minos_lower = ""
                minos_upper = ""

        print(
            f"{i:3d} | {p.name:14s} | {p.value:12.6f} | {err_str:>12s} | "
            f"{minos_lower:>12} | {minos_upper:>12} | "
            f"{str(p.is_fixed):>5s} | {str(low):10s} | {str(high):10s}"
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

            print(
                f"{'':>3} | {'ε_sig':14s} | {sig_val:12.6f} | {err_col:>12s} | "
                f"{lo_col:>12} | {hi_col:>12} | {fixed_flag:>5s} | {'':10s} | {'':10s}"
            )

    print(sep)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_fit_progress(m, sigmoid_eff=False):
    eps = m.values['epsilon']
    if sigmoid_eff:
        eps = sigmoid(eps)
    print(f"Iteration: N={m.values['N']:.1f}, ε={eps:.3f}, "
          f"B_p={m.values['B_p']:.1f}, B_f={m.values['B_f']:.1f}, "
          f"fval={m.fval:.1f}")
    
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
    nfcn = getattr(m, "nfcn", None)  

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
        (valid_min,              "Valid Minimum",               "INVALID Minimum"),
        (not any_at_limit,       "No parameters at limit",      "Some parameters at limit"),
        (not above_edm,          "Below EDM threshold",         "Above EDM threshold"),
        (not reached_call_limit, "Below call limit",            "Reached call limit"),
        (not hesse_failed,       "Hesse ok",                    "Hesse failed"),
        (cov_accurate,           "Covariance ok",               "Covariance APPROXIMATE"),
    ]

    summary = []
    # 4) Print the combined summary box
    summary.append("\n" + "#"*60)
    summary.append("\n" + "="*60)
    # Title
    summary.append(f"Fit Summary: {args_data}, {args_bin}, {fit_type}")
    summary.append("="*60)
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
        except:
            pass
    summary.append(f"Fit Success   : {'Yes' if m.valid else 'No'}")
    summary.append("=" * 60)

    # 6) Fit Parameters table, can add more if needed
    summary.append("Fit Parameters:")
    if 'N' in popt:
        summary.append(f"  Total N          = {popt['N']:.1f} ± {perr.get('N', float('nan')):.1f}")
    if 'epsilon' in popt:
        if sigmoid_eff:
            eff = sigmoid(popt['epsilon'])
            eff_err = abs(perr.get('epsilon', float('nan')) * eff * (1 - eff))
            summary.append(f"  Efficiency ε     = {eff:.6f} ± {eff_err:.10f} (sigmoid)")
        else:
            summary.append(f"  Efficiency ε     = {popt['epsilon']:.6f} ± {perr.get('epsilon', float('nan')):.10f}")
    if 'B_p' in popt:
        summary.append(f"  Background B_p   = {popt['B_p']:.1f} ± {perr.get('B_p', float('nan')):.1f}")
    if 'B_f' in popt:
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
    if 'epsilon' in popt:
        try:
            lo, hi = edges_pass[0], edges_pass[-1]
            if sigmoid_eff:
                eff = sigmoid(popt['epsilon'])
                eff_err = abs(perr.get('epsilon', float('nan')) * eff * (1 - eff))
                summary.append("Efficiency:")
                summary.append(f"  ε = {eff:.4f} ± {eff_err:.4f} (bin {lo:.1f} - {hi:.1f} GeV, sigmoid)")
            else:
                summary.append("Efficiency:")
                summary.append(f"  ε = {popt['epsilon']:.4f} ± {perr.get('epsilon', float('nan')):.4f} (bin {lo:.1f} - {hi:.1f} GeV)")
        except Exception:
            if sigmoid_eff:
                eff = sigmoid(popt['epsilon'])
                eff_err = abs(perr.get('epsilon', float('nan')) * eff * (1 - eff))
                summary.append(f"Efficiency: ε = {eff:.4f} ± {eff_err:.4f} (sigmoid)")
            else:
                summary.append(f"Efficiency: ε = {popt['epsilon']:.4f} ± {perr.get('epsilon', float('nan')):.4f}")
    # Bottom border
    summary.append("="*60 + "\n")
    summary.append("#"*60)

    return "\n".join(summary)
    
class PassFailPlotter:
    def __init__(self, cost_func_pass, error_func_pass, cost_func_fail, error_func_fail, n_bins_pass, edges_pass, edges_fail, fit_type, sigmoid_eff = False):
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
        self.signal_param_names = SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
        self.bg_param_names = BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
        self.sigmoid_eff = sigmoid_eff
        plt.rcParams.update({'font.size': 8})  # sets global font size

    def __call__(self, args):
        param_dict = dict(zip(self.param_names, args))

        # Split the data
        data_pass = self.cost.data
        data_fail = self.cost_fail.data
        data_pass = data_pass[:self.n_bins_pass]
        data_fail = data_fail[:self.n_bins_pass]

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

        signal_y_pass = (N - (B_p + B_f))* epsilon * signal_pass
        signal_y_fail = (N - (B_p + B_f))* (1 - epsilon) * signal_fail
        bg_y_pass = B_p * bg_pass
        bg_y_fail = B_f * bg_fail
        total_pass = signal_y_pass + bg_y_pass

        # model yields → densities
        signal_y_pass = signal_y_pass * widths_pass
        bg_y_pass     = bg_y_pass     * widths_pass
        total_pass    = total_pass    * widths_pass

        total_fail = signal_y_fail + bg_y_fail

        signal_y_fail = signal_y_fail * widths_fail
        bg_y_fail     = bg_y_fail     * widths_fail
        total_fail    = total_fail    * widths_fail

        # Plot pass
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("Pass")
        plt.errorbar(cx_pass, data_pass,   yerr=self.error_pass, fmt='o', color='black', label='Data')
        plt.stairs(bg_y_pass, self.edges_pass, fill=True, color='orange', label='Background')
        plt.stairs(total_pass, self.edges_pass, baseline=bg_y_pass, fill=True, color='skyblue', label='Signal')
        plt.stairs(total_pass, self.edges_pass, color='navy', label='Total Fit')
        plt.legend()

        # Plot fail
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Fail")
        plt.errorbar(cx_fail, data_fail, yerr=self.error_fail, fmt='o', color='black', label='Data')
        plt.stairs(bg_y_fail, self.edges_fail, fill=True, color='orange', label='Background')
        plt.stairs(total_fail, self.edges_fail, baseline=bg_y_fail, fill=True, color='skyblue', label='Signal')
        plt.stairs(total_fail, self.edges_fail, color='navy', label='Total Fit')
        plt.legend()

        plt.tight_layout()

class CombinedCost:
    def __init__(self, cost1, cost2):
        self.cost1 = cost1
        self.cost2 = cost2
        self.ndata = cost1.ndata + cost2.ndata

    def __call__(self, *params):
        return self.cost1(*params) + self.cost2(*params)

def load_histogram(root_file, hist_name, data_label):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            is_mc = ("MC" in data_label) or ("MC" in hist_name)
            print(f"Histogram: {hist_name}")
            return {"values": values, "edges": edges, "errors": obj.errors(), "is_mc": is_mc}
    return None

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

def double_crystal_ball_pdf(x, mu, sigma, alphaL, nL, alphaR, nR):
    nL = np.clip(nL, 1, 100)
    nR = np.clip(nR, 1, 100)

    z = (x - mu) / sigma
    result = np.zeros_like(z)

    # avoid division by zero
    abs_aL = max(np.abs(alphaL), 1e-8)
    abs_aR = max(np.abs(alphaR), 1e-8)

    # core
    core = np.exp(-0.5 * z**2)
    mask_core = (z > -abs_aL) & (z < abs_aR)
    result[mask_core] = core[mask_core]

    # left tail
    mask_L = z <= -abs_aL
    # log of normalization constant
    logNL = nL * np.log(nL/abs_aL) - 0.5 * abs_aL**2
    tL = (nL/abs_aL - abs_aL - z[mask_L])
    tL = np.maximum(tL, 1e-8)
    result[mask_L] = np.exp(logNL - nL * np.log(tL))

    # right tail
    mask_R = z >= abs_aR
    logNR = nR * np.log(nR/abs_aR) - 0.5 * abs_aR**2
    tR = (nR/abs_aR - abs_aR + z[mask_R])
    tR = np.maximum(tR, 1e-8)
    result[mask_R] = np.exp(logNR - nR * np.log(tR))

    # final normalization
    norm = np.trapezoid(result, x)
    if norm <= 0 or not np.isfinite(norm):
        norm = 1e-8
    return result / norm 

def double_crystal_ball_cdf(x, mu, sigma, alphaL, nL, alphaR, nR):
    betaL = alphaL; mL = nL; scaleL = sigma; betaR = alphaR; mR = nR; scaleR = sigma; loc = mu
    def cdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
        x = np.asarray(x)
        T = type(beta_left)
        sqrt_half = np.sqrt(T(0.5))
        sqrt_pi = np.sqrt(T(np.pi))
        # Left tail constants
        exp_bl = np.exp(-T(0.5) * beta_left * beta_left)
        a_bl = (m_left / beta_left) ** m_left * exp_bl
        b_bl = m_left / beta_left - beta_left
        m1_bl = m_left - T(1)
        # Right tail constants
        exp_br = np.exp(-T(0.5) * beta_right * beta_right)
        a_br = (m_right / beta_right) ** m_right * exp_br
        b_br = m_right / beta_right - beta_right
        m1_br = m_right - T(1)

        # Normalization
        norm = (a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(0.0) - special.erf(-beta_left * sqrt_half))) * scale_left + (
                a_br * (b_br + beta_right) ** -m1_br / m1_br + sqrt_half * sqrt_pi * (special.erf(beta_right * sqrt_half) - special.erf(0.0))) * scale_right
        r = np.empty_like(x)

        for i in range(len(x)):
            scale = T(1) / (scale_left if x[i] < loc else scale_right)
            z = (x[i] - loc) * scale
            if z < -beta_left:
                r[i] = a_bl * (b_bl - z) ** -m1_bl / m1_bl * scale_left / norm
            elif z < 0:
                r[i] = (a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(z * sqrt_half) - special.erf(-beta_left * sqrt_half))) * scale_left / norm
            elif z < beta_right:
                r[i] = (a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(0.0) - special.erf(-beta_left * sqrt_half))
                        ) * scale_left + sqrt_half * sqrt_pi * (special.erf(z * sqrt_half) - special.erf(0.0)) * scale_right
                r[i] /= norm
            else:
                r[i] = (
                    a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(0.0) - special.erf(-beta_left * sqrt_half))) * scale_left + (sqrt_half * 
                    sqrt_pi * (special.erf(beta_right * sqrt_half) - special.erf(0.0)) + a_br * (b_br + beta_right) ** -m1_br / m1_br - a_br * (b_br + z) ** -m1_br / m1_br) * scale_right
                r[i] /= norm
        return r
    
    cb1 = cdf(x, betaL, mL, scaleL, betaR, mR, scaleR, loc)

    return cb1

def double_voigtian(x, mu, sigma1, gamma1, sigma2, gamma2):
    result = (voigt_profile(x-mu, sigma1, gamma1) + 
              voigt_profile(x-mu, sigma2, gamma2))
    # Normalize
    return result / np.trapezoid(result, x)

def gaussian_pdf(x, mu, sigma):
    # normalized Gaussian
    return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def gaussian_cdf(x, mu, sigma):
    return (1/2 * (1 + special.erf((x - mu) / (np.sqrt(2) * sigma))))

def CB_G(x, mu, sigma, alpha, n, sigma2):
    def crystal_ball_unnormalized(x, mu, sigma, alpha, n):
        z = (x - mu) / sigma
        result = np.zeros_like(z)
        abs_alpha = np.abs(alpha)

        # Core region (Gaussian)
        if alpha < 0:
            mask_core = z > -abs_alpha
            mask_tail = z <= -abs_alpha
        else:
            mask_core = z < abs_alpha
            mask_tail = z >= abs_alpha

        result[mask_core] = np.exp(-0.5 * z[mask_core]**2)

        # Tail region (Power law)
        # Calculate N safely using log sum exp trick
        try:
            logN = n * np.log(n / abs_alpha) - 0.5 * abs_alpha**2
            N = np.exp(logN)
        except FloatingPointError:
            N = 1e300  # fallback large number

        base = (n / abs_alpha - abs_alpha - z[mask_tail]) if (alpha < 0) else (n / abs_alpha - abs_alpha + z[mask_tail])
        base = np.clip(base, 1e-15, np.inf)  # prevent zero or negative values

        result[mask_tail] = N * base**(-n)
        return result

    y_cb_un = crystal_ball_unnormalized(x, mu, sigma, alpha, n)

    # Normalize
    integral = np.trapezoid(y_cb_un, x)
    y_cb = y_cb_un / integral

    y_gauss = norm.pdf(x, loc=mu, scale=sigma2)

    y_total = y_cb + y_gauss
    normalization = np.trapezoid(y_total, x)
    if normalization <= 0 or np.isnan(normalization) or np.isinf(normalization):
        return np.zeros_like(y_total)
    return y_total / normalization

def phase_space(x, a, b, x_min=x_min, x_max=x_max):
    # Clip exponents into a safe range
    a_clamped = np.clip(a, 0, 20)
    b_clamped = np.clip(b, 0, 20)

    # 2) Work in log‐space
    t1 = np.clip(x - x_min, 1e-8, None)
    t2 = np.clip(x_max - x, 1e-8, None)

    log_pdf = a_clamped * np.log(t1) + b_clamped * np.log(t2)
    pdf = np.exp(log_pdf - np.max(log_pdf))   # subtract max for stability

    # zero outside
    pdf[(x <= x_min) | (x >= x_max)] = 0

    # Normalize
    norm = np.trapezoid(pdf, x)
    return pdf / (norm if norm>0 else 1e-8)

def linear_pdf(x, b, C, x_min=x_min, x_max=x_max):
    x_mid = 0.5 * (x_min + x_max)
    lin = b * (x - x_mid) + C

    # Clip negative values
    lin = np.clip(lin, 0, None)

    denom = np.trapezoid(lin, x)

    return lin / denom

def linear_cdf(x, b, C, x_min=x_min, x_max=x_max):
    x = np.asarray(x)
    den = 0.5*b*(x_max**2 - x_min**2) + C*(x_max - x_min)
    if den <= 0:
        den = 1e-8
    cdf = np.zeros_like(x, dtype=float)
    mask = (x > x_min) & (x < x_max)
    if np.any(mask):
        xm = x[mask]
        num = 0.5*b*(xm**2 - x_min**2) + C*(xm - x_min)
        cdf[mask] = num / den
    cdf[x >= x_max] = 1.0
    return cdf

def exponential_pdf(x, C, x_min=x_min, x_max=x_max):
    z = -C * x
    z_max = np.max(z)
    # subtract z_max to stabilize
    exp_z = np.exp(z - z_max)
    # normalize using log-sum-exp
    log_norm = z_max + np.log(np.trapezoid(exp_z, x))
    norm = np.exp(log_norm)

    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(x)

    return np.exp(z) / norm
    
def exponential_cdf(x, C, x_min=x_min, x_max=x_max):
    cdf = (1 - np.exp(-C*x))
    return cdf

def chebyshev_background(x, *coeffs, x_min=x_min, x_max=x_max):
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)

def bernstein_poly(x, *coeffs, x_min = x_min, x_max = x_max):
    c = np.array(coeffs).reshape(-1, 1)
    return BPoly(c, [x_min, x_max])(x)

def cms(x, beta, gamma, loc):
    y = cmsshape.pdf(x, beta, gamma, loc)
    return y

def create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=False, sigmoid_eff=False):
    config = FIT_CONFIGS[fit_type]

    # If either CDF is missing, fall back to PDF mode for the entire region
    if use_cdf and (config["signal_cdf"] is None or config["background_cdf"] is None):
        use_cdf = False

    signal_func = config["signal_cdf"] if use_cdf else config["signal_pdf"]
    bg_func     = config["background_cdf"] if use_cdf else config["background_pdf"]
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
    signal_params = [params_dict[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]
    
    # Background parameters
    bg_pass_params = [params_dict[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
    bg_fail_params = [params_dict[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
    
    ## Normalize signal and background components
    bin_widths_pass = np.diff(edges_pass)
    bin_widths_fail = np.diff(edges_fail)

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
    
    result_pass = ((N - (B_p + B_f)) * epsilon * signal_pass + B_p * bg_pass)
    result_fail = ((N - (B_p + B_f)) * (1 - epsilon) * signal_fail + B_f * bg_fail)

    result_pass = np.clip(result_pass, 1e-10, None)
    result_fail = np.clip(result_fail, 1e-10, None)
    return result_pass, result_fail 

def calculate_custom_chi2(values, errors, model, n_params):
    mask = (errors > 0) & (model > 0) & (values >= 0)
    
    # Calculate Pearson chi2
    Pearson_chi2 = np.sum(((values[mask] - model[mask]) / errors[mask])**2)
    
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

def fit_function(fit_type, hist_pass, hist_fail, fixed_params=None, use_cdf=False, x_min=x_min, x_max=x_max, interactive=False, args_bin=None, args_data=None, sigmoid_eff=False, args_mass=None):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]

    if use_cdf and (config["signal_cdf"] is None or config["background_cdf"] is None):
        print(f"[Warning] Model '{fit_type}' missing CDF(s). Disabling CDF mode.")
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

    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
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
            N_p0 = (np.sum(values_pass) + np.sum(values_fail))
            B_p_p0 = max(1, np.median(values_pass[-10:]) * len(values_pass))
            B_f_p0 = max(1, np.median(values_fail[-10:]) * len(values_fail))

            # Scale fixed parameters if present
            for name in ['N', 'epsilon', 'B_p', 'B_f']:
                if name in fixed_params:
                    fixed_params[name]

            # Update bounds with data-based values
            bounds = config["bounds"].copy()
            bounds.update({
                "N":       (B_p_p0 + B_f_p0, N_p0, np.inf),
                "B_p":     (0, B_p_p0/4, np.inf),
                "B_f":     (0, B_f_p0, np.inf),
            })

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
        for name in ['N', 'epsilon', 'B_p', 'B_f']:
            if name in fixed_params:
                fixed_params[name]

        # Update bounds with data-based values
        bounds = config["bounds"].copy()
        bounds.update({
            "N":       (0, N_p0, np.inf),
            "B_p":     (0, B_p_p0, np.inf),
            "B_f":     (0, B_f_p0, np.inf),
        })


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
        result_pass, _ = create_combined_model(fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_approx_fail(edges, *params):
        _, result_fail = create_combined_model(fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_fail

    def model_cdf_pass(edges, *params):
        result_pass, _ = create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_cdf_fail(edges, *params):
        _, result_fail = create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_fail

    bin_widths_pass = np.diff(edges_pass)
    bin_widths_fail = np.diff(edges_fail)

    if use_cdf:
        model_pass = model_cdf_pass
        model_fail = model_cdf_fail
    else:
        model_pass = model_approx_pass
        model_pass = model_pass
        model_fail = model_approx_fail
        model_fail = model_fail

    # Cost functions depending on if using CDF or PDF
    if use_cdf:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass)
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail)
        c_fail.errdef = Minuit.LIKELIHOOD
    else:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass, use_pdf='approximate')
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail, use_pdf='approximate')
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
    plotter = PassFailPlotter(c_pass, errors_pass, c_fail, errors_fail, len(edges_pass), edges_pass, edges_fail, fit_type, sigmoid_eff=sigmoid_eff)
    if interactive:
        m.interactive(plotter)
    else:
        # RUN FITTER HERE
        m.simplex()
        m.migrad()

    for param in m.parameters:
        try:
            m.minos(param)
        except Exception as e:
            print(f"MINOS failed for parameter {param}: {str(e)}")

    # Print results
    m.print_level = 0
    print_minuit_params_table(m, sigmoid_eff=sigmoid_eff)
    print_fit_progress(m, sigmoid_eff=sigmoid_eff)

    # 2. Extract results
    fcv = m.fval
    popt = m.values.to_dict()
    perr = m.errors.to_dict()
    cov = m.covariance

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
    summary_text = print_fit_summary(m, popt, perr, edges_pass, args_bin, BINS_INFO, Pearson_chi2, Poisson_chi2, total_ndof, args_data, fit_type, sigmoid_eff=sigmoid_eff)
    
    print(summary_text)

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
            minos_errors[param] = {
                'lower': m.merrors[param].lower,
                'upper': m.merrors[param].upper
            }
        else:
            minos_errors[param] = {
                'lower': -m.errors[param],  # Fall back to symmetric errors
                'upper': m.errors[param]
            }

    # Return results
    results = {
        "param_names": param_names,
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
    }

    results["summary"] = summary_text
    results["bin"] = args_bin
    return results
    
def plot_combined_fit(results, plot_dir=".", data_type="DATA", fixed_params=None, sigmoid_eff=False, args_abseta=None, args_mass=None):
    if results is None:
        print("No results to plot")
        return None, None  # Return None if no results

    fixed_params = fixed_params or {}
    fit_type = results["type"]
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_pdf"]
    bg_func = config["background_pdf"]
    params = results["popt"]
    perr = results["perr"]

    # Get signal and background model names for the legend
    signal_model_name = {
        "dcb": "Double Crystal Ball",
        "dv": "Double Voigtian",
        "g": "Gaussian",
        "cb_g": "Crystal Ball + Gaussian"
    }.get(fit_type.split('_')[0], "Unknown Signal")

    background_model_name = {
        "ps": "Phase Space",
        "lin": "Linear",
        "exp": "Exponential",
        "cheb": "Chebyshev Polynomial",
        "cms": "CMS Shape",
        "bpoly": "Bernstein Polynomial"
    }.get(fit_type.split('_')[1], "Unknown Background")

    x = np.linspace(results["x_min"], results["x_max"], 1000)
    signal_params = [params[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]

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

    bg_pass_params = [params[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
    signal_pass =((params["N"] - (params["B_p"] + params["B_f"])) * eff * signal_func(x, *signal_params))
    bg_pass = (params["B_p"] * bg_func(x, *bg_pass_params))
    total_pass = (signal_pass + bg_pass)

    signal_pass = signal_pass * np.mean(bin_widths_pass)
    bg_pass = bg_pass * np.mean(bin_widths_pass)
    total_pass = total_pass * np.mean(bin_widths_pass)

    ax_pass.errorbar(
        results["centers_pass"],
        results["values_pass"],
        yerr=results["errors_pass"],
        fmt="o", markersize=6, capsize=3,
        color="royalblue", label="Data (Pass)")
    ax_pass.plot(x, total_pass, 'k-', label="Total fit")
    ax_pass.plot(x, signal_pass, 'g--', label=f"Signal ({signal_model_name})")
    ax_pass.plot(x, bg_pass, 'r--', label=f"Background ({background_model_name})")

    ax_pass.set_xlabel("$m_{ee}$ [GeV]", fontsize=12)
    ax_pass.set_ylabel("Events / GeV", fontsize=12)
    bin_suffix, bin_range = BINS_INFO[results['bin']]
    ax_pass.set_title(f"{data_type.replace('_', ' ')}: {bin_range} GeV (Pass)", pad=10)

    chi2_red = results["reduced_chi_squared"]
    signal_params_text = "\n".join([
        format_param(p, params[p], results["perr"][p], fixed_params)
        for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
    ])
    bg_params_text = "\n".join([
        format_param(f"{p}_pass", params[f"{p}_pass"], results["perr"][f"{p}_pass"], fixed_params)
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
    ])

    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {eff:.6f} ± {eff_err:.6f}{eff_label}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        f"",
        f"Converged: {results['converged']}",
        f""
        f"\nSignal yield: {params['N']*eff:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"NLL: χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
        f"Pearson: χ²/ndof = {results['Pearson_chi2']:.1f}/{results['total_ndof']} = {results['Pearson_tot_red_chi2']:.2f}",
        f"Poisson: χ²/ndof = {results['Poisson_chi2']:.1f}/{results['total_ndof']} = {results['Poisson_tot_red_chi2']:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text
    ]
    ax_pass.legend(loc="upper right", fontsize=10)
    ax_pass.text(
        0.02, 0.98,
        "\n".join(info_text),
        transform=ax_pass.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )

    os.makedirs(plot_dir, exist_ok=True)
    if args_mass == "Z":
        fig_pass.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Pass.png", 
                    bbox_inches="tight", dpi=300)
    elif args_mass == "JPsi":
            fig_pass.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Pass.png", 
                bbox_inches="tight", dpi=300)
    elif args_mass == "JPsi_muon":
        fig_pass.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_abseta{args_abseta}_Pass.png",
                        bbox_inches="tight", dpi=300)
    elif args_mass == "Z_muon":
        fig_pass.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_abseta{args_abseta}_Pass.png",
                    bbox_inches="tight", dpi=300)

    # --- FAIL plot ---
    fig_fail, ax_fail = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")

    bg_fail_params = [params[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
    signal_fail = ((params["N"] - (params["B_p"] + params["B_f"])) * (1-eff) * signal_func(x, *signal_params))
    bg_fail = (params["B_f"] * bg_func(x, *bg_fail_params))
    total_fail = (signal_fail + bg_fail)

    signal_fail = signal_fail * np.mean(bin_widths_fail)
    bg_fail = bg_fail * np.mean(bin_widths_fail)
    total_fail = total_fail * np.mean(bin_widths_fail)

    ax_fail.errorbar(
        results["centers_fail"],
        results["values_fail"],
        yerr=results["errors_fail"],
        fmt="o", markersize=6, capsize=3,
        color="royalblue", label="Data (Fail)")
    ax_fail.plot(x, total_fail, 'k-', label="Total fit")
    ax_fail.plot(x, signal_fail, 'purple', linestyle = '--', label=f"Signal ({signal_model_name})")
    ax_fail.plot(x, bg_fail, 'r--', label=f"Background ({background_model_name})")

    ax_fail.set_xlabel("$m_{ee}$ [GeV]", fontsize=12)
    ax_fail.set_ylabel("Events / GeV", fontsize=12)
    bin_suffix, bin_range = BINS_INFO[results['bin']]
    ax_fail.set_title(f"{data_type.replace('_', ' ')}: {bin_range} GeV (Fail)", pad=10)
    bg_params_text_fail = "\n".join([
        format_param(f"{p}_fail", params[f"{p}_fail"], results["perr"][f"{p}_fail"], fixed_params)
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
    ])
    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {eff:.6f} ± {eff_err:.6f}{eff_label}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        f"",
        f"Converged: {results['converged']}",
        f""
        f"\nSignal yield: {params['N']*(1-eff):.1f}",
        f"Bkg yield: {params['B_f']:.1f}",
        f"NLL: χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
        f"Pearson: χ²/ndof = {results['Pearson_chi2']:.1f}/{results['total_ndof']} = {results['Pearson_tot_red_chi2']:.2f}",
        f"Poisson: χ²/ndof = {results['Poisson_chi2']:.1f}/{results['total_ndof']} = {results['Poisson_tot_red_chi2']:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text_fail
    ]
    ax_fail.legend(loc="upper right", fontsize=10)
    ax_fail.text(
        0.02, 0.98,
        "\n".join(info_text),
        transform=ax_fail.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )

    if args_mass == "Z":
        fig_fail.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Fail.png", 
                    bbox_inches="tight", dpi=300)
    elif args_mass == "JPsi":
            fig_fail.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Fail.png", 
                bbox_inches="tight", dpi=300)
    elif args_mass == "JPsi_muon":
        fig_fail.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_abseta{args_abseta}_Fail.png",
                        bbox_inches="tight", dpi=300)
    elif args_mass == "Z_muon":
        fig_fail.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_abseta{args_abseta}_Fail.png",
                    bbox_inches="tight", dpi=300)
    print(f"Plots saved to {plot_dir}")

    return fig_pass, fig_fail

def get_bin_info(mass):
    if mass == "Z":
        return {
            f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(5,7), (7,10), (10,20), (20,45), (45,75), (75,500)])  
        }
    elif mass == "Z_muon":
        return {
            f"bin{i+1}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(5,7), (7,10), (10,20), (20,45), (45,75), (75,500)])  
        }
    elif mass == "JPsi":
        return {
            f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(5, 7), (7, 10), (10, 20)])
        }  
    elif mass == "JPsi_muon":
        return {
            f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
            for i, (lo, hi) in enumerate([(3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,30)])
        }  

BINS_INFO = get_bin_info(mass)

def shape_params(mass):
    if mass == "Z" or mass == "Z_muon":
        SIGNAL_MODELS = {
        "dcb": {
            "pdf": double_crystal_ball_pdf,
            "cdf": double_crystal_ball_cdf,
            "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
            "bounds": {
                "mu": (88, 90.5, 92),
                "sigma": (1, 3, 6),
                "alphaL": (0, 1.0, 10),
                "nL": (0, 5.0, 30),
                "alphaR": (0, 1.0, 10),
                "nR": (0, 5.0, 30)
            }
        },
        "dv": {
            "pdf": double_voigtian,
            "cdf": None,
            "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
            "bounds": {
                "mu": (88, 90, 93),
                "sigma1": (2.0, 3.0, 4.0),
                "gamma1": (0.01, 0.5, 3.0),
                "sigma2": (1.0, 2.0, 3.0),
                "gamma2": (0.01, 1.0, 3.0)
            }
        },
        "g": {
            "pdf": gaussian_pdf,
            "cdf": gaussian_cdf,
            "params": ["mu", "sigma"],
            "bounds": {
                "mu": (88, 90, 94),
                "sigma": (1, 2.5, 6)
            }
        },
        "cbg": {
            "pdf": CB_G,
            "cdf": None,
            "params": ["mu", "sigma", "alpha", "n", "sigma2"],
            "bounds": {
                "mu": (88, 90, 92),
                "sigma": (1, 3, 6),
                "alpha": (-10, -1, 10),
                "n": (0, 5.0, 30),
                "sigma2": (1, 3, 10)
            }
        }
    }

        BACKGROUND_MODELS = {
            "ps": {
                "pdf": lambda x, a, b: phase_space(x, a, b, x_min=x_min, x_max=x_max),
                "cdf": None, 
                "params": ["a", "b"],
                "bounds": {
                    "a": (0, 0.5, 10),
                    "b": (0, 1, 30)
                }
            },
            "lin": {
                "pdf": linear_pdf,
                "cdf": linear_cdf,
                "params": ["b", "C"],
                "bounds": {
                    "b": (-1, 0.1, 1),
                    "C": (0, 0.1, 10)
                }
            },
            "exp": {
                "pdf": exponential_pdf,
                "cdf": exponential_cdf,
                "params": ["C"],
                "bounds": {
                    "C": (-10, 0.1, 10)
                }
            },
            "cheb": {
                "pdf": chebyshev_background,
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0.001, 1, 3),
                    "c1": (0.001, 1, 3),
                    "c2": (0.001, 1, 3)
                }
            },
            "bpoly": {
                "pdf": bernstein_poly,
                "cdf": None,        
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0, 0.05, 10),
                    "c1": (0, 0.1, 1),
                    "c2": (0, 0.1, 1),
                }
            },
            "cms": {
                "pdf": cms,
                "cdf": None,
                "params": ["beta", "gamma", "loc"],
                "bounds": {
                    "beta": (-0.5, 0.1, 1.5),
                    "gamma": (0, 0.1, 2),   
                    "loc": (-100, 90, 200)     
                }
            }
        }

    elif mass == "JPsi_muon" or mass == "JPsi":
        SIGNAL_MODELS = {
            "dcb": {
                "pdf": double_crystal_ball_pdf,
                "cdf": double_crystal_ball_cdf,
                "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
                "bounds": {
                    "mu": (2.5, 3.05, 3.5),
                    "sigma": (0, 0.03, 0.06),
                    "alphaL": (0, 1.0, 10),
                    "nL": (0, 5.0, 30),
                    "alphaR": (0, 1.0, 10),
                    "nR": (0, 5.0, 30)
                }
            },
            "dv": {
                "pdf": double_voigtian,
                "cdf": None,
                "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
                "bounds": {
                    "mu": (2.5, 3.05, 3.5),
                    "sigma1": (0, 0.03, 0.06),
                    "gamma1": (0.01, 0.5, 3.0),
                    "sigma2": (1.0, 2.0, 3.0),
                    "gamma2": (0.01, 1.0, 3.0)
                }
            },
            "g": {
                "pdf": gaussian_pdf,
                "cdf": gaussian_cdf,
                "params": ["mu", "sigma"],
                "bounds": {
                    "mu": (2.5, 3.05, 3.5),
                    "sigma": (0, 0.03, 0.1)
                }
            },
            "cbg": {
                "pdf": CB_G,
                "cdf": None,
                "params": ["mu", "sigma", "alpha", "n", "sigma2"],
                "bounds": {
                    "mu": (2.5, 3.05, 3.5),
                    "sigma": (0, 0.03, 0.06),
                    "alpha": (-10, -1, 10),
                    "n": (0.1, 5.0, 30),
                    "sigma2": (0, 0.03, 0.15)
                }
            }
        }

        BACKGROUND_MODELS = {
            "ps": {
                "pdf": lambda x, a, b: phase_space(x, a, b, x_min=x_min, x_max=x_max),
                "cdf": None, 
                "params": ["a", "b"],
                "bounds": {
                    "a": (0, 0.5, 10),
                    "b": (0, 1, 30)
                }
            },
            "lin": {
                "pdf": linear_pdf,
                "cdf": linear_cdf,
                "params": ["b", "C"],
                "bounds": {
                    "b": (-1, 0.1, 1),
                    "C": (0, 0.1, 10)
                }
            },
            "exp": {
                "pdf": exponential_pdf,
                "cdf": exponential_cdf,
                "params": ["C"],
                "bounds": {
                    "C": (-10, 0.1, 10)
                }
            },
            "cheb": {
                "pdf": chebyshev_background,
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0.001, 1, 3),
                    "c1": (0.001, 1, 3),
                    "c2": (0.001, 1, 3)
                }
            },
            "bpoly": {
                "pdf": bernstein_poly,
                "cdf": None,        
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0, 0.05, 10),
                    "c1": (0, 0.1, 1),
                    "c2": (0, 0.1, 1),
                }
            },
            "cms": {
                "pdf": cms,
                "cdf": None,
                "params": ["beta", "gamma", "loc"],
                "bounds": {
                    "beta": (-0.5, 0.1, 10),
                    "gamma": (0, 0.1, 10),   
                    "loc": (-100, 3, 100)     
                }
            }
        }

    return SIGNAL_MODELS, BACKGROUND_MODELS

SIGNAL_MODELS, BACKGROUND_MODELS = shape_params(mass)

FIT_CONFIGS = {}
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
        bounds = {
            "N": (0, 100000, np.inf),
            "epsilon": (0, 0.9, 1),
            "B_p": (0, 10000, np.inf),
            "B_f": (0, 10000, np.inf)
        }
        
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
