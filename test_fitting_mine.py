import uproot
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse
from scipy.special import wofz
from scipy.special import voigt_profile
from numpy.polynomial.chebyshev import Chebyshev

def load_histogram(root_file, hist_name):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            errors = obj.errors()  #  use stored errors, not sqrt(N)
            return {"values": values, "edges": edges, "errors": errors}
    return None

def create_fixed_param_wrapper(func, fixed_params: dict):
    def wrapped(x, *free_params):
        full_params = []
        free_index = 0
        total_params = len(fixed_params) + len(free_params)
        for i in range(total_params):
            if i in fixed_params:
                full_params.append(fixed_params[i])
            else:
                full_params.append(free_params[free_index])
                free_index += 1
        return func(x, *full_params)
    return wrapped


def double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR):
    z = (x - mu) / sigma
    result = np.zeros_like(z)

    mask_core = (z > -alphaL) & (z < alphaR)
    result[mask_core] = A * np.exp(-0.5 * z[mask_core]**2)

    mask_left = z <= -alphaL
    abs_alphaL = np.abs(alphaL)
    NL = (nL / abs_alphaL)**nL * np.exp(-0.5 * abs_alphaL**2)
    result[mask_left] = A * NL * (nL / abs_alphaL - abs_alphaL - z[mask_left])**(-nL)

    mask_right = z >= alphaR
    abs_alphaR = np.abs(alphaR)
    NR = (nR / abs_alphaR)**nR * np.exp(-0.5 * abs_alphaR**2)
    result[mask_right] = A * NR * (nR / abs_alphaR - abs_alphaR + z[mask_right])**(-nR)

    return result

def double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2):
    voigt1 = A * voigt_profile(x - mu, sigma1, gamma1)
    voigt2 = A * voigt_profile(x - mu, sigma2, gamma2)
    return voigt1 + voigt2

#def double_voigtian(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R):
#    def voigt_profile(x, mu, sigma, gamma):
#        z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
#        return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))                            
#    V_left = voigt_profile(x, mu, sigma_L, gamma_L)
#    V_right = voigt_profile(x, mu, sigma_R, gamma_R)
#    shape = A * (eta * V_left + (1 - eta) * V_right)
#    return shape

def double_gaussian(x, A, mu, sigma_1, sigma_2):
    def gaussian(x, mu, sigma):
        z = ((x - mu) / sigma)
        return np.exp(-0.5 * z**2) / sigma * np.sqrt(2 * np.pi)
    g_1 = A * gaussian(x, mu, sigma_1)
    g_2 = A * gaussian(x, mu, sigma_2)
    return g_1 + g_2
    

def phase_space(x, B, a, b, x_min, x_max):
    delta = 1e-5  # or 1e-3 depending on bin resolution
    safe_x = np.clip(x, x_min + delta, x_max - delta)
    shape = (safe_x - x_min)**a * (x_max - safe_x)**b
    shape[(x <= x_min) | (x >= x_max)] = 0
    return B * shape

def linear(x, B, C):
    shape = B * (1 + C * x)
    return shape

def exponential(x, B, C):
    shape = B * np.e**(C * x)
    return shape

def chebyshev_background(x, *coeffs, x_min=70, x_max=110):
    """
    Chebyshev polynomial background function.
    
    Args:
        x: Input values
        *coeffs: Chebyshev coefficients (c0, c1, c2, ...)
        x_min, x_max: Range for normalization (default 70, 110)
    
    Returns:
        Background values at points x
    """
    # Normalize x to [-1, 1] interval (important for Chebyshev polynomials)
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Create and evaluate Chebyshev polynomial
    poly = Chebyshev(coeffs)
    return poly(x_norm)

def dcb_plus_phase(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b, x_min, x_max):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + phase_space(x, B, a, b, x_min, x_max)
def dcb_plus_linear(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + linear(x, B, C)
def dcb_plus_exponential(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + exponential(x, B, C)
def dcb_plus_cheb(x, A, mu, sigma, alphaL, nL, alphaR, nR, *coeffs, x_min, x_max):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + chebyshev_background(x, *coeffs, x_min=x_min, x_max=x_max)
def dv_plus_phase(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, a, b, x_min, x_max):
    return double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2) + phase_space(x, B, a, b, x_min, x_max)
def dv_plus_linear(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C):
    return double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2) + linear(x, B, C)
def dv_plus_exponential(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C):
    return double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2) + exponential(x, B, C)
def dv_plus_cheb(x, A, mu, sigma1, gamma1, sigma2, gamma2, *coeffs, x_min, x_max):
    return double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2) + chebyshev_background(x, *coeffs, x_min = x_min, x_max=x_max)
def dg_plus_phase(x, A, mu, sigma_1, sigma_2, B, a, b, x_min, x_max):
    return double_gaussian(x, A, mu, sigma_1, sigma_2) + phase_space(x, B, a, b, x_min, x_max)
def dg_plus_linear(x, A, mu, sigma_1, sigma_2, B, C):
    return double_gaussian(x, A, mu, sigma_1, sigma_2) + linear(x, B, C)
def dg_plus_exponential(x, A, mu, sigma_1, sigma_2, B, C):
    return double_gaussian(x, A, mu, sigma_1, sigma_2) + exponential(x, B, C)
def dg_plus_cheb(x, A, mu, sigma_1, sigma_2, *coeffs, x_min, x_max):
    return double_gaussian(x, A, mu, sigma_1, sigma_2) + chebyshev_background(x, *coeffs, x_min=x_min, x_max=x_max)

def compute_signal_background_events_dcb_ps(x, popt, x_min, x_max):
    signal = double_crystal_ball(x, *popt[:7])
    background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dcb_lin(x, popt):
    signal = double_crystal_ball(x, *popt[:7])
    background = linear(x, popt[7], popt[8])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dcb_exp(x, popt):
    signal = double_crystal_ball(x, *popt[:7])
    background = exponential(x, popt[7], popt[8])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dcb_cheb(x, popt, x_min, x_max):
    signal = double_crystal_ball(x, *popt[:7])
    background = chebyshev_background(x, *popt[7:], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_ps(x, popt, x_min, x_max):
    signal = double_voigtian(x, *popt[:6])
    background = phase_space(x, popt[6], popt[7], popt[8], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_lin(x, popt):
    signal = double_voigtian(x, *popt[:6])
    background = linear(x, popt[6], popt[7])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_exp(x, popt):
    signal = double_voigtian(x, *popt[:6])
    background = exponential(x, popt[6], popt[7])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_cheb(x, popt, x_min, x_max):
    signal = double_voigtian(x, *popt[:6])
    background = chebyshev_background(x, *popt[6:], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dg_ps(x, popt, x_min, x_max):
    signal = double_gaussian(x, *popt[:4])
    background = phase_space(x, popt[4], popt[5], popt[6], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dg_lin(x, popt):
    signal = double_gaussian(x, *popt[:4])
    background = linear(x, popt[4], popt[5])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dg_exp(x, popt):
    signal = double_gaussian(x, *popt[:4])
    background = exponential(x, popt[5], popt[5])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dg_cheb(x, popt, x_min, x_max):
    signal = double_gaussian(x, *popt[:4])
    background = chebyshev_background(x, *popt[4:], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)

def perform_fit(type, hist, hist_name, fixed_params=None):
    print(f"Fitting histogram '{hist_name}' ...")
    
    if fixed_params is None:
        fixed_params = {}
    
    centers = (hist["edges"][:-1] + hist["edges"][1:]) / 2
    values = hist["values"]
    errors = hist["errors"]
    errors[errors == 0] = 1.0  # avoid zero division

    x_min, x_max = 70, 110
    mask = (centers >= x_min) & (centers <= x_max)
    centers = centers[mask]
    values = values[mask]
    errors = errors[mask]
    
    # Define parameter bounds and initial guesses for each fit type
    FIT_CONFIGS = {
        "dcb_ps": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "a", "b"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma": (2.75, 2.76, 2.77),
                "alphaL": (0.9, 0.94, 0.95),
                "nL": (0.1, 5, 30),
                "alphaR": (1.78, 1.79, 1.9),
                "nR": (0.1, 5, 30),
                "B": (0.000001, 100, np.inf),
                "a": (0.1, 1.44, 2.45),
                "b": (0.1, 2.09, 4)
            }
        },
        "dcb_lin": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values), np.inf),
                "mu": (89, 90, 91),
                "sigma": (1, 2.76, 4),
                "alphaL": (0.1, 2, 1000000),
                "nL": (0.1, 5, 60),
                "alphaR": (0.1, 2, 10),
                "nR": (0.1, 5, 1000000),
                "B": (0, 0.5, 5),
                "C": (-2, -0.5, 0)
            }
        },
        "dcb_exp": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
            "bounds": {
                "A": (max(values)*0.1, max(values)*0.8, np.inf),
                "mu": (89, 90, 91),
                "sigma": (2.75, 2.76, 2.77),
                "alphaL": (0.93, 0.94, 0.95),
                "nL": (0.1, 5, 30),
                "alphaR": (1.78, 1.79, 1.8),
                "nR": (0.1, 5, 30),
                "B": (0.00001, 1, np.inf),
                "C": (0.0001, 0.1, 10)
            }
        },
        "dcb_cheb": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR"] + [f"c{i}" for i in range(3)],
            "bounds": {
                "A": (max(values)*0.5, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma": (2, 2.76, 2.77),
                "alphaL": (0.9, 0.94, 0.95),
                "nL": (0.1, 5, 30),
                "alphaR": (1.78, 1.79, 1.9),
                "nR": (0.1, 5, 30),
                "c0": (-30, 0, 30),
                "c1": (-30, 0, 30),
                "c2": (-30, -5,-2)
            }
        },
        "dv_ps": {
            "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "a", "b"],
            "bounds": {
                "A": (max(values)*0.1, max(values)*5, np.inf),
                "mu": (89, 90, 91),
                "sigma1": (3.23, 3.24, 3.25),
                "gamma1": (0.01, 0.02, 0.03),
                "sigma2": (2.08, 2.09, 2.10),
                "gamma2": (0.81, 0.82, 0.83),
                "B": (0.000001, 1, np.inf),
                "a": (0.1, 2, 3),
                "b": (0.1, 2, 4)
            }
        },
        "dv_lin": {
            "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma1": (0.1, 2.5, 8),
                "gamma1": (0.1, 0.7, 8),
                "sigma2": (0.1, 2.5, 8),
                "gamma2": (0.1, 0.7, 8),
                "B": (-1, 1, np.inf),
                "C": (-1, 0.5, 0)
            }
        },
        "dv_exp": {
            "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, np.inf),
                "mu": (89, 90, 92),
                "sigma1": (3.23, 3.24, 3.9),
                "gamma1": (0.01, 0.2, 3),
                "sigma2": (2.11, 2.12, 2.13),
                "gamma2": (0.90, 0.91, 0.92),
                "B": (0.001, 1, np.inf),
                "C": (-10, 0.1, 10)
            }
        },
        "dv_cheb": {
            "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2"] + [f"c{i}" for i in range(3)],
            "bounds": {
                "A": (max(values)*0.5, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma1": (2, 2.5, 3),
                "gamma1": (1, 2, 3),
                "sigma2": (2, 2.5, 3),
                "gamma2": (1, 2, 3),
                "c0": (0, 1, 10),
                "c1": (-10, 0, 10),
                "c2": (-10, 0, 10)
            }
     },
        "dg_ps": {
            "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "a", "b"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma_1": (0.5, 2, 7),
                "sigma_2": (0.5, 2, 7),
                "B": (0.000001, 100, np.inf),
                "a": (0.1, 1.44, 2.45),
                "b": (0.1, 2.09, 4)
            }
        },
        "dg_lin": {
            "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, np.inf),
                "mu": (89, 90, 91),
                "sigma_1": (2.75, 2.76, 2.77),
                "sigma_2": (0.93, 0.94, 0.95),
                "B": (0.000001, 1, np.inf),
                "C": (-0.5, 0, 0.5)
            }
        },
        "dg_exp": {
            "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "C"],
            "bounds": {
                "A": (max(values)*0.1, max(values)*0.8, np.inf),
                "mu": (89, 90, 91),
                "sigma": (2.75, 2.76, 2.77),
                "sigma_2": (0.93, 0.94, 0.95),
                "B": (0.00001, 1, np.inf),
                "C": (0.0001, 0.1, 10)
            }
        },
        "dg_cheb": {
            "param_names": ["A", "mu", "sigma_1", "sigma_2"] + [f"c{i}" for i in range(3)],
            "bounds": {
                "A": (max(values)*0.2, max(values)*2, np.inf),
                "mu": (89, 90, 91),
                "sigma_1": (2, 2.76, 6),
                "sigma_2": (0.9, 0.94, 3),
                "c0": (-30, 0, 30),
                "c1": (-30, 0, 30),
                "c2": (-30, -5,-2)
            }
        },
    }

    def get_fit_parameters(fit_type):
        """Get initial parameters and bounds for the specified fit type"""
        config = FIT_CONFIGS[fit_type]
        
        lower_dict = {}
        p0_dict = {}
        upper_dict = {}
        
        for param, (lower, p0, upper) in config["bounds"].items():
            lower_val = lower
            p0_val = p0
            upper_val = upper
                
            lower_dict[param] = lower_val
            p0_dict[param] = p0_val
            upper_dict[param] = upper_val
        
        return config["param_names"], lower_dict, p0_dict, upper_dict

    # Get all parameters first
    param_names, lower_dict, p0_dict, upper_dict = get_fit_parameters(type)
    
    # Create lists of free parameters (excluding fixed ones)
    free_param_names = [name for name in param_names if name not in fixed_params]
    free_p0 = [p0_dict[name] for name in free_param_names]
    
    # Adjust bounds for free parameters only
    free_lower = [lower_dict[name] for name in free_param_names]
    free_upper = [upper_dict[name] for name in free_param_names]

    # Create a mapping of parameter indices to names for the fixed parameters wrapper
    param_index_map = {i: name for i, name in enumerate(param_names)}
    fixed_params_indices = {i: fixed_params[name] for i, name in param_index_map.items() if name in fixed_params}

# DOUBLE CRYSTAL BALL PLUS PHASE SPACE/CHEBYSHEV
    if type == "dcb_ps" or type == "dcb_cheb":
        if type == "dcb_ps":
            def full_model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b):
                return dcb_plus_phase(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b, x_min, x_max)
        elif type == "dcb_cheb":
            def full_model(x, A, mu, sigma, alphaL, nL, alphaR, nR, *coeffs):
                return dcb_plus_cheb(x, A, mu, sigma, alphaL, nL, alphaR, nR, *coeffs, x_min=x_min, x_max=x_max)
# DOUBLE CRYSTAL BALL PLUS LINEAR/EXPONENTIAL
    elif type == "dcb_lin" or type == "dcb_exp":
        if type == "dcb_lin":
            def full_model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
                return dcb_plus_linear(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C)
        elif type == "dcb_exp":
            def full_model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
                return dcb_plus_exponential(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C)
# DOUBLE VOIGTIAN PLUS PHASE SPACE/CHEBYSHEV
    elif type == "dv_ps" or type == "dv_cheb":
        if type == "dv_ps":
            def full_model(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, a, b):
                return dv_plus_phase(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, a, b, x_min, x_max)
        elif type == "dv_cheb":
            def full_model(x, A, mu, sigma1, gamma1, sigma2, gamma2, *coeffs):
                return dv_plus_cheb(x, A, mu, sigma1, gamma1, sigma2, gamma2, *coeffs, x_min=x_min, x_max=x_max)
# DOUBLE VOIGTIAN PLUS LINEAR/EXPONENTIAL
    elif type == "dv_lin" or type == "dv_exp":
        if type == "dv_lin":
            def full_model(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C):
                return dv_plus_linear(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C)
        elif type == "dv_exp":
            def full_model(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C):
                return dv_plus_exponential(x, A, mu, sigma1, gamma1, sigma2, gamma2, B, C)
# DOUBLE GAUSSIAN PLUS PHASE SPACE/CHEBYSHEV
    elif type == "dg_ps" or type == "dg_cheb":
        if type == "dg_ps":
            def full_model(x, A, mu, sigma_1, sigma_2, B, a, b):
                return dg_plus_phase(x, A, mu, sigma_1, sigma_2, B, a, b, x_min, x_max)
        elif type == "dg_cheb":
            def full_model(x, A, mu, sigma_1, sigma_2, *coeffs):
                return dg_plus_cheb(x, A, mu, sigma_1, sigma_2, *coeffs, x_min=x_min, x_max=x_max)
# DOUBLE GAUSSIAN PLUS LINEAR/EXPONENTIAL
    elif type == "dg_lin" or type == "dg_exp":
        if type == "dg_lin":
            def full_model(x, A, mu, sigma_1, sigma_2, B, C):
                return dg_plus_linear(x, A, mu, sigma_1, sigma_2, B, C)
        elif type == "dg_exp":
            def full_model(x, A, mu, sigma_1, sigma_2, B, C):
                return dg_plus_exponential(x, A, mu, sigma_1, sigma_2, B, C)

        # Create the wrapped model with fixed parameters
    
    # Create the wrapped model with fixed parameters
    if fixed_params_indices:
        model = create_fixed_param_wrapper(full_model, fixed_params_indices)
    else:
        model = full_model

    # Perform the fit with only free parameters
    popt, pcov, infodict, errmsg, ier = curve_fit(
        model, centers, values, p0=free_p0, sigma=errors,
        absolute_sigma=True, bounds=(free_lower, free_upper), full_output=True, maxfev=20000)
    
    if ier == 1 or ier == 2:
        print(f"Curve fit converged successfully, ier = {ier}")
    elif ier == 3 or ier == 4 or ier == 5:
        print(f"Curve fit did not converge, ier = {ier}")
        print("Error message:", errmsg)

    #perr = np.sqrt(np.diag(pcov))

    # Reconstruct the full parameter list including fixed parameters
    full_popt = []
    full_perr = np.zeros(len(param_names))
    free_index = 0
    for i, name in enumerate(param_names):
        if name in fixed_params:
            full_popt.append(fixed_params[name])
            full_perr[i] = 0.0  # error is zero for fixed parameters
        else:
            full_popt.append(popt[free_index])
            if pcov is not None and free_index < len(pcov):
                full_perr[i] = np.sqrt(np.diag(pcov))[free_index] if pcov.ndim == 2 else np.sqrt(pcov[free_index])
            else:
                full_perr[i] = 0.0
            free_index += 1

    # Verify fixed parameters were actually fixed
    for i, name in enumerate(param_names):
        if name in fixed_params:
            if not np.isclose(full_popt[i], fixed_params[name], atol=1e-6):
                print(f"Warning: Fixed parameter {name} changed from {fixed_params[name]} to {full_popt[i]}")
                full_popt[i] = fixed_params[name]  # Force the fixed value

    # Calculate signal and background events
    x = np.linspace(x_min, x_max, 1000)
    if type == "dcb_ps":
        signal_events, background_events = compute_signal_background_events_dcb_ps(x, full_popt, x_min, x_max)
    elif type == "dcb_lin":
        signal_events, background_events = compute_signal_background_events_dcb_lin(x, full_popt)
    elif type == "dcb_exp":
        signal_events, background_events = compute_signal_background_events_dcb_exp(x, full_popt)
    elif type == "dcb_cheb":
        signal_events, background_events = compute_signal_background_events_dcb_cheb(x, full_popt, x_min, x_max)
    elif type == "dv_ps":
        signal_events, background_events = compute_signal_background_events_dv_ps(x, full_popt, x_min, x_max)
    elif type == "dv_lin":
        signal_events, background_events = compute_signal_background_events_dv_lin(x, full_popt)
    elif type == "dv_exp":
        signal_events, background_events = compute_signal_background_events_dv_exp(x, full_popt)
    elif type == "dv_cheb":
        signal_events, background_events = compute_signal_background_events_dv_cheb(x, full_popt, x_min, x_max)
    elif type == "dg_ps":
        signal_events, background_events = compute_signal_background_events_dg_ps(x, full_popt, x_min, x_max)
    elif type == "dg_lin":
        signal_events, background_events = compute_signal_background_events_dg_lin(x, full_popt)
    elif type == "dg_exp":
        signal_events, background_events = compute_signal_background_events_dg_exp(x, full_popt)
    elif type == "dg_cheb":
        signal_events, background_events = compute_signal_background_events_dg_cheb(x, full_popt, x_min, x_max)

    # Calculate errors
    dA, dsigma = full_perr[0], full_perr[2]
    signal_error = np.sqrt((full_popt[2] * np.sqrt(2 * np.pi) * dA)**2 + (full_popt[0] * np.sqrt(2 * np.pi) * dsigma)**2)

    if type == "dcb_ps":
        dB = full_perr[7]
        background_shape = (x - x_min)**popt[8] * (x_max - x)**popt[9]
        background_error = np.trapezoid(background_shape, x) * dB
    elif type == "dcb_lin" or type == "dcb_exp":
        background_error = np.sqrt(full_perr[8]**2)
    elif type == "dcb_cheb":
        background_error = np.sqrt(np.sum(full_perr[7:]**2)) * (x_max - x_min)
    elif type == "dv_ps":
        dB = full_perr[6]
        background_shape = (x - x_min)**popt[7] * (x_max - x)**popt[8]
        background_error = np.trapezoid(background_shape, x) * dB
    elif type == "dv_lin" or type == "dv_exp":
        background_error = np.sqrt(full_perr[7]**2)
    elif type == "dv_cheb":
        background_error = np.sqrt(np.sum(full_perr[6:]**2)) * (x_max - x_min)
    elif type == "dg_ps":
        dB = full_perr[4]
        background_shape = (x - x_min)**popt[5] * (x_max - x)**popt[6]
        background_error = np.trapezoid(background_shape, x) * dB
    elif type == "dg_lin" or type == "dg_exp":
        background_error = np.sqrt(full_perr[5]**2)
    elif type == "dg_cheb":
        background_error = np.sqrt(np.sum(full_perr[4:]**2)) * (x_max - x_min)

    expected = model(centers, *popt)
    residuals = values - expected
    standardized_residuals = residuals / errors

    chi_squared = np.sum(((values - expected) / errors) ** 2)
    dof = len(values) - len(popt)
    reduced_chi_squared = chi_squared / dof

    print(f" - Estimated number of signal events: {signal_events:.2f} ± {signal_error:.2f}")
    print(f" - Estimated number of background events: {background_events:.2f} ± {background_error:.2f}")
    print(f" - Chi-squared: {chi_squared:.2f}, Reduced Chi-squared: {reduced_chi_squared:.2f}")

    fit_results = {
        'centers': centers,
        'values': values,
        'errors': errors,
        'x_min': x_min,
        'x_max': x_max,
        'popt': full_popt,
        'perr': full_perr,
        'signal_events': signal_events,
        'signal_error': signal_error,
        'background_events': background_events,
        'background_error': background_error,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'type': type,
        'hist_name': hist_name,
        'ier': ier
    }
    
    return fit_results

FIT_CONFIGS = {
    "dcb_ps": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "a", "b"],
        "bounds": {
            "A": (10, 100000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2.75, 2.76, 2.77),
            "alphaL": (0.9, 0.94, 0.95),
            "nL": (0.1, 5, 30),
            "alphaR": (1.78, 1.79, 1.9),
            "nR": (0.1, 5, 30),
            "B": (0.000001, 100, np.inf),
            "a": (0.1, 1.44, 2.45),
            "b": (0.1, 2.09, 4)
        }
    },
    "dcb_lin": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
        "bounds": {
            "A": (100000, 800000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (1, 2.76, 4),
            "alphaL": (0.1, 2, 1000000),
            "nL": (0.1, 5, 60),
            "alphaR": (0.1, 2, 10),
            "nR": (0.1, 5, 1000000),
            "B": (0, 0.5, 5),
            "C": (-2, -0.5, 0)
        }
    },
    "dcb_exp": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
        "bounds": {
            "A": (10, 100000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2.75, 2.76, 2.77),
            "alphaL": (0.93, 0.94, 0.95),
            "nL": (0.1, 5, 30),
            "alphaR": (1.78, 1.79, 1.8),
            "nR": (0.1, 5, 30),
            "B": (0.00001, 1, np.inf),
            "C": (0.0001, 0.1, 10)
        }
    },
    "dcb_cheb": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR"] + [f"c{i}" for i in range(3)],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2, 2.76, 2.77),
            "alphaL": (0.9, 0.94, 0.95),
            "nL": (0.1, 5, 30),
            "alphaR": (1.78, 1.79, 1.9),
            "nR": (0.1, 5, 30),
            "c0": (-30, 0, 30),
            "c1": (-30, 0, 30),
            "c2": (-30, -5,-2)
        }
    },
    "dv_ps": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "a", "b"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (3.23, 3.24, 3.25),
            "gamma1": (0.01, 0.02, 0.03),
            "sigma2": (2.08, 2.09, 2.10),
            "gamma2": (0.81, 0.82, 0.83),
            "B": (0.000001, 1, np.inf),
            "a": (0.1, 2, 3),
            "b": (0.1, 2, 4)
        }
    },
    "dv_lin": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (3.23, 3.24, 3.9),
            "gamma1": (0.66, 0.67, 0.68),
            "sigma2": (2.11, 2.12, 2.13),
            "gamma2": (0.90, 0.91, 0.92),
            "B": (0.001, 1, np.inf),
            "C": (-0.8, 0.1, 0.8)
        }
    },
    "dv_exp": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 92),
            "sigma1": (3.23, 3.24, 3.9),
            "gamma1": (0.01, 0.2, 3),
            "sigma2": (2.11, 2.12, 2.13),
            "gamma2": (0.90, 0.91, 0.92),
            "B": (0.001, 1, np.inf),
            "C": (-10, 0.1, 10)
        }
    },
    "dv_cheb": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2"] + [f"c{i}" for i in range(3)],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (2, 2.5, 3),
            "gamma1": (1, 2, 3),
            "sigma2": (2, 2.5, 3),
            "gamma2": (1, 2, 3),
            "c0": (0, 1, 10),
            "c1": (-10, 0, 10),
            "c2": (-10, 0, 10)
        }
    },
    "dg_ps": {
        "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "a", "b"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma_1": (0.5, 2, 7),
            "sigma_2": (0.5, 2, 7),
            "B": (0.000001, 100, np.inf),
            "a": (0.1, 1.44, 2.45),
            "b": (0.1, 2.09, 4)
        }
    },
    "dg_lin": {
        "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma_1": (2.75, 2.76, 2.77),
            "sigma_2": (0.93, 0.94, 0.95),
            "B": (0.000001, 1, np.inf),
            "C": (-0.5, 0, 0.5)
        }
    },
    "dg_exp": {
        "param_names": ["A", "mu", "sigma_1", "sigma_2", "B", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2.75, 2.76, 2.77),
            "sigma_2": (0.93, 0.94, 0.95),
            "B": (0.00001, 1, np.inf),
            "C": (0.0001, 0.1, 10)
        }
    },
    "dg_cheb": {
        "param_names": ["A", "mu", "sigma_1", "sigma_2"] + [f"c{i}" for i in range(3)],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma_1": (2, 2.76, 6),
            "sigma_2": (0.9, 0.94, 3),
            "c0": (-30, 0, 30),
            "c1": (-30, 0, 30),
            "c2": (-30, -5,-2)
        }
    },
}

def plot_fit(fit_results, efficiency=None, efficiency_error=None, plot_dir=".", data_type="DATA", Npass=None, Nfail=None):
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")

    centers = fit_results['centers']
    values = fit_results['values']
    errors = fit_results['errors']
    x_min = fit_results['x_min']
    x_max = fit_results['x_max']
    popt = fit_results['popt']
    perr = fit_results['perr']
    type = fit_results['type']
    hist_name = fit_results['hist_name']
    ier = fit_results['ier']

    plt.errorbar(centers, values, yerr=errors, fmt='o', color='royalblue', capsize=3)

    x = np.linspace(x_min, x_max, 1000)

    if type == "dcb_ps":
        signal = double_crystal_ball(x, *popt[:7])
        background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
        signal_events, background_events = compute_signal_background_events_dcb_ps(x, popt, x_min, x_max)
        signal_label = "Double Crystal Ball"
        background_label = "Phase-space"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Phase Background: B={popt[7]:.5f} ± {perr[7]:.5f}\n a={popt[8]:.2f} ± {perr[8]:.2f}, b={popt[9]:.2f} ± {perr[9]:.2f}"
    elif type == "dcb_lin":
        signal = double_crystal_ball(x, *popt[:7])
        background = linear(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dcb_lin(x, popt)
        signal_label = "Double Crystal Ball"
        background_label = "Linear"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Linear Background: B={popt[7]:.5f} ± {perr[7]:.5f}\n C={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dcb_exp":
        signal = double_crystal_ball(x, *popt[:7])
        background = exponential(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dcb_exp(x, popt)
        signal_label = "Double Crystal ball"
        background_label = "Exponential"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Exp Background: B={popt[7]:.5f} ± {perr[7]:.5f}\n C={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dcb_cheb":
        signal = double_crystal_ball(x, *popt[:7])
        background = chebyshev_background(x, *popt[7:], x_min=x_min, x_max=x_max)
        signal_events, background_events = compute_signal_background_events_dcb_cheb(x, popt, x_min, x_max)
        signal_label = "Double Crystal Ball"
        background_label = "Chebyshev"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = "Chebyshev Background: " + ", ".join([f"c{i}={popt[7+i]:.3f} ± {perr[7+i]:.3f}" for i in range(len(popt)-7)])
    elif type == "dv_ps":
        signal = double_voigtian(x, *popt[:6])
        background = phase_space(x, popt[6], popt[7], popt[8], x_min, x_max)
        signal_events, background_events = compute_signal_background_events_dv_ps(x, popt, x_min, x_max)
        signal_label = "Double Voigtian"
        background_label = "Phase-space"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n γ1={popt[3]:.2f} ± {perr[3]:.2f}, σ2={popt[4]:.2f} ± {perr[4]:.2f}, γ2={popt[5]:.2f} ± {perr[5]:.2f}\n"
        background_params = f"Phase Background: B={popt[6]:.5f} ± {perr[6]:.5f}\n a={popt[7]:.2f} ± {perr[7]:.2f}, b={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dv_lin":
        signal = double_voigtian(x, *popt[:6])
        background = linear(x, popt[6], popt[7])
        signal_events, background_events = compute_signal_background_events_dv_lin(x, popt)
        signal_label = "Double Voigtian"
        background_label = "Linear"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n γ1={popt[3]:.2f} ± {perr[3]:.2f}, σ2={popt[4]:.2f} ± {perr[4]:.2f}, γ2={popt[5]:.2f} ± {perr[5]:.2f}\n"
        background_params = f"Linear Background: B={popt[6]:.5f} ± {perr[6]:.5f}\n C={popt[7]:.2f} ± {perr[7]:.2f}"
    elif type == "dv_exp":
        signal = double_voigtian(x, *popt[:6])
        background = exponential(x, popt[6], popt[7])
        signal_events, background_events = compute_signal_background_events_dv_exp(x, popt)
        signal_label = "Double Voigtian"
        background_label = "Exponential"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n γ1={popt[3]:.2f} ± {perr[3]:.2f}, σ2={popt[4]:.2f} ± {perr[4]:.2f}, γ2={popt[5]:.2f} ± {perr[5]:.2f}\n"
        background_params = f"Exponential Background: B={popt[6]:.5f} ± {perr[6]:.5f}\n C={popt[7]:.2f} ± {perr[7]:.2f}"
    elif type == "dv_cheb":
        signal = double_voigtian(x, *popt[:6])
        background = chebyshev_background(x, *popt[6:], x_min=x_min, x_max=x_max)
        signal_events, background_events = compute_signal_background_events_dv_cheb(x, popt, x_min, x_max)
        signal_label = "Double Voigtian"
        background_label = "Chebyshev"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n γ1={popt[3]:.2f} ± {perr[3]:.2f}, σ2={popt[4]:.2f} ± {perr[4]:.2f}, γ2={popt[5]:.2f} ± {perr[5]:.2f}\n"
        background_params = "Chebyshev Background: " + ", ".join([f"c{i}={popt[6+i]:.3f} ± {perr[6+i]:.3f}" for i in range(len(popt)-6)])
    elif type == "dg_ps":
        signal = double_gaussian(x, *popt[:4])
        background = phase_space(x, popt[4], popt[5], popt[6], x_min, x_max)
        signal_events, background_events = compute_signal_background_events_dg_ps(x, popt, x_min, x_max)
        signal_label = "Double Gaussian"
        background_label = "Phase-space"
        signal_params = f"DG (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n σ2={popt[3]:.2f} ± {perr[3]:.2f}\n"
        background_params = f"Phase Background: B={popt[4]:.5f} ± {perr[4]:.5f}\n a={popt[5]:.2f} ± {perr[5]:.2f}, b={popt[6]:.2f} ± {perr[6]:.2f}"
    elif type == "dg_lin":
        signal = double_gaussian(x, *popt[:4])
        background = linear(x, popt[4], popt[5])
        signal_events, background_events = compute_signal_background_events_dg_lin(x, popt)
        signal_label = "Double Gausian"
        background_label = "Linear"
        signal_params = f"DG (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n σ2={popt[3]:.2f} ± {perr[3]:.2f}\n"
        background_params = f"Linear Background: B={popt[4]:.5f} ± {perr[4]:.5f}\n C={popt[5]:.2f} ± {perr[5]:.2f}"
    elif type == "dg_exp":
        signal = double_gaussian(x, *popt[:4])
        background = exponential(x, popt[4], popt[5])
        signal_events, background_events = compute_signal_background_events_dg_exp(x, popt)
        signal_label = "Double Gaussian"
        background_label = "Exponential"
        signal_params = f"DG (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n σ2={popt[3]:.2f} ± {perr[3]:.2f}\n"
        background_params = f"Exponential Background: B={popt[4]:.5f} ± {perr[4]:.5f}\n C={popt[4]:.2f} ± {perr[4]:.2f}"
    elif type == "dg_cheb":
        signal = double_gaussian(x, *popt[:4])
        background = chebyshev_background(x, *popt[4:], x_min=x_min, x_max=x_max)
        signal_events, background_events = compute_signal_background_events_dg_cheb(x, popt, x_min, x_max)
        signal_label = "Double Gaussian"
        background_label = "Chebyshev"
        signal_params = f"DG (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ1={popt[2]:.2f} ± {perr[2]:.2f}\n σ2={popt[3]:.2f} ± {perr[3]:.2f}\n"
        background_params = "Chebyshev Background: " + ", ".join([f"c{i}={popt[4+i]:.3f} ± {perr[4+i]:.3f}" for i in range(len(popt)-4)])

    combined = signal + background

    plt.plot(x, signal, color='orange', label=signal_label)
    plt.plot(x, background, color='red', label=background_label)
    plt.plot(x, combined, color='black', label="Total Fit")

    if ier == 1 or ier == 2:
        convergence = f"Converges, ier: {ier}"
    elif ier == 3 or ier == 4 or ier == 5:
        convergence = f"Does NOT Converge, ier: {ier}"
    
    # Build legend text
    legend_text = (
        f"{signal_params}"
        f"{background_params}\n"
        f"Signal Events: {fit_results['signal_events']:.2f} ± {fit_results['signal_error']:.2f}\n"
        f"Background Events: {fit_results['background_events']:.2f} ± {fit_results['background_error']:.2f}\n"
        f"χ² = {fit_results['chi_squared']:.2f}, Reduced χ² = {fit_results['reduced_chi_squared']:.2f}"
        f"\nEfficiency: {efficiency:.4f} ± {efficiency_error:.8f}\n"
        f"Npass: {Npass:.3f}, Nfail: {Nfail:.3f}\n"
        f"{convergence}"
    )
    
    plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"Fit to {hist_name} ({background_label} background)")
    plt.xlabel(r"$m_{ee}$ [GeV]")
    plt.ylabel("Number of events")
    plt.legend(loc='upper right', fontsize=8)
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{data_type}_{type}_fit_{hist_name}.svg")
    plt.close()
    print(f"Plot saved for {hist_name}\n")

def main():
    # ... (parser setup, bins definition, pass_fixed_params parsing, file loading - unchanged) ...
    bins = {
        "bin00": ("pt_5p00To8p00", "5.00-8.00"),
        "bin01": ("pt_8p00To10p00", "8.00-10.00"),
        "bin02": ("pt_10p00To15p00", "10.00-15.00"),
        "bin03": ("pt_15p00To20p00", "15.00-20.00"),
        "bin04": ("pt_20p00To30p00", "20.00-30.00"),
        "bin05": ("pt_30p00To35p00", "30.00-35.00"),
        "bin06": ("pt_35p00To40p00", "35.00-40.00"),
        "bin07": ("pt_40p00To45p00", "40.00-45.00"),
        "bin08": ("pt_45p00To50p00", "45.00-50.00"),
        "bin09": ("pt_50p00To55p00", "50.00-55.00"),
        "bin10": ("pt_55p00To60p00", "55.00-60.00"),
        "bin11": ("pt_60p00To80p00", "60.00-80.00"),
        "bin12": ("pt_80p00To100p00", "80.00-100.00"),
        "bin13": ("pt_100p00To150p00", "100.00-150.00"),
        "bin14": ("pt_150p00To250p00", "150.00-250.00"),
        "bin15": ("pt_250p00To400p00", "250.00-400.00")
    }

    parser = argparse.ArgumentParser(description="Fit ROOT histograms with different models.")
    parser.add_argument("--bin", type=str, required=True, choices=bins.keys(),
                       help="Which bin to fit (e.g., bin00, bin01, etc.)")
    parser.add_argument("--type", type=str, required=True,
                       choices=["dcb_ps", "dcb_lin", "dcb_exp", "dcb_cheb", "dv_ps", "dv_lin", "dv_exp", "dv_cheb", "dg_ps", "dg_lin", "dg_exp", "dg_cheb"])
    parser.add_argument("--data", type=str, required=True, choices=["DATA_barrel_1", "DATA_barrel_2", "MC_barrel_1", "MC_barrel_2"])
    parser.add_argument("--fix", type=str, default="",
                       help="Comma-separated list of parameters to fix FOR THE PASS FIT ONLY in format param1=value1,param2=value2")
    args = parser.parse_args()

    pass_fixed_params = {}
    if args.fix:
        for item in args.fix.split(','):
            if '=' in item:
                key_val_pair = item.split('=', 1)
                if len(key_val_pair) == 2:
                    param_name = key_val_pair[0].strip()
                    value_str = key_val_pair[1].strip()
                    try:
                        pass_fixed_params[param_name] = float(value_str)
                    except ValueError:
                        print(f"Warning: Could not convert value '{value_str}' for PASS FIT parameter '{param_name}'. Ignoring.")
                else:
                    print(f"Warning: Ignoring malformed fixed parameter segment for PASS FIT '{item}'")
            else:
                print(f"Warning: Ignoring malformed fixed parameter segment for PASS FIT '{item}' (missing '=')")

    if args.data == "DATA_barrel_1":
        root_file_path = "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_2_23D_histos_pt_barrel_1.root"
    elif args.data == "DATA_barrel_2":
        root_file_path = "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_2_23D_histos_pt_barrel_2.root"
    elif args.data == "MC_barrel_1":
        root_file_path = "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_2_23D_histos_pt_barrel_1.root"
    elif args.data == "MC_barrel_2":
        root_file_path = "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_2_23D_histos_pt_barrel_2.root"
    else:
        print(f"Error: Unknown data type '{args.data}'")
        return

    try:
        root_file = uproot.open(root_file_path)
    except FileNotFoundError:
        print(f"Error: ROOT file not found at {root_file_path}")
        return
    except Exception as e:
        print(f"Error opening ROOT file {root_file_path}: {e}")
        return

    if args.data.startswith("DATA"):
        plot_dir = f"{args.bin}_fits/DATA/"
    else:
        plot_dir = f"{args.bin}_fits/MC/"
    os.makedirs(plot_dir, exist_ok=True)

    bin_suffix, bin_range = bins[args.bin]
    hist_name_pass = f"{args.bin}_{bin_suffix}_Pass"
    hist_name_fail = f"{args.bin}_{bin_suffix}_Fail"

    hist_pass = load_histogram(root_file, hist_name_pass)
    hist_fail = load_histogram(root_file, hist_name_fail)

    if hist_pass is None:
        print(f"Error: Could not load Pass histogram '{hist_name_pass}' from {root_file_path}")
        root_file.close()
        return
    if hist_fail is None:
        print(f"Error: Could not load Fail histogram '{hist_name_fail}' from {root_file_path}")
        root_file.close()
        return

    # --- 1. FIT PASS HISTOGRAM ---
    print(f"\n--- Fitting PASS histogram: {hist_name_pass} ---")
    fit_results_pass = perform_fit(args.type, hist_pass, hist_name_pass, pass_fixed_params)

    if fit_results_pass['ier'] not in [1, 2]:
        print(f"Warning: Fit for PASS histogram {hist_name_pass} did not converge well (ier={fit_results_pass['ier']}).")

    # --- 2. PREPARE FIXED PARAMETERS FOR FAIL FIT ---
    fail_fixed_params = {}
    popt_pass = fit_results_pass['popt']

    if args.type not in FIT_CONFIGS: # FIT_CONFIGS should be globally defined
        print(f"Error: FIT_CONFIGS not defined for type '{args.type}'. Cannot proceed.")
        root_file.close()
        return
        
    all_param_names_for_type = FIT_CONFIGS[args.type]["param_names"]
    
    # Define parameters that should ALWAYS float in the Fail fit
    params_to_float_in_fail = ["A"] # Signal amplitude 'A' always floats

    # For Chebyshev backgrounds, 'c0' (primary amplitude) should float
    if args.type.endswith("_cheb"):
        params_to_float_in_fail.append("c0")
    # Note: If the model is not _cheb, the parameter 'B' (if it exists for _ps, _lin, _exp)
    # will NOT be added to params_to_float_in_fail by default, thus it will be fixed.

    # If model is DCB-based, add nL and nR to floating parameters for Fail fit
    if args.type.startswith("dcb_"):
        params_to_float_in_fail.extend(["nL", "nR"])

    # Build the fail_fixed_params dictionary
    for i, param_name in enumerate(all_param_names_for_type):
        if param_name not in params_to_float_in_fail:
            fail_fixed_params[param_name] = popt_pass[i]

    print("\nParameters to be fixed for FAIL fit (from PASS fit results):")
    for p, v in fail_fixed_params.items():
        print(f"  {p} = {v:.4f}")
    
    floating_list_str = ", ".join([f"'{p}'" for p in params_to_float_in_fail])
    print(f"Floating parameters for FAIL fit: {floating_list_str}")


    # --- 3. FIT FAIL HISTOGRAM ---
    print(f"\n--- Fitting FAIL histogram: {hist_name_fail} (some shapes from PASS) ---")
    fit_results_fail = perform_fit(args.type, hist_fail, hist_name_fail, fail_fixed_params)

    if fit_results_fail['ier'] not in [1, 2]:
        print(f"Warning: Fit for FAIL histogram {hist_name_fail} did not converge well (ier={fit_results_fail['ier']}).")

    # --- Calculate efficiency ---
    Npass = fit_results_pass['signal_events']
    Npass_err = fit_results_pass['signal_error']
    Nfail = fit_results_fail['signal_events']
    Nfail_err = fit_results_fail['signal_error']

    efficiency = Npass / (Npass + Nfail)
    efficiency_error = np.sqrt(efficiency * (1 - efficiency) / (Npass + Nfail))
    
    print(f"\nEfficiency for pt bin {bin_range} GeV = {efficiency:.4f} ± {efficiency_error:.4f}")
    print(f"Npass = {Npass:.2f} +/- {Npass_err:.2f}")
    print(f"Nfail = {Nfail:.2f} +/- {Nfail_err:.2f}")

    # --- Plotting ---
    # Make sure plot_fit function signature matches how it's called.
    # The one from your last script was:
    # plot_fit(fit_results, efficiency=None, efficiency_error=None, plot_dir=".", data_type="DATA", Npass=None, Nfail=None)
    plot_fit(fit_results_pass, efficiency, efficiency_error, plot_dir, f"{args.data}", Npass=Npass, Nfail=Nfail)
    plot_fit(fit_results_fail, efficiency, efficiency_error, plot_dir, f"{args.data}", Npass=Npass, Nfail=Nfail)


    root_file.close()

if __name__ == "__main__":
    # Ensure FIT_CONFIGS (defined globally before main) is correctly populated 
    # with all types and their bounds for all parameters.
    main()

# 108976904.90 
# Err: 3595203.46
# FRAC: 0.0032988