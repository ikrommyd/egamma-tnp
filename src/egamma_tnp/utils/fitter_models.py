from __future__ import annotations

from functools import partial

import numpy as np
from numba_stats import cmsshape
from numpy.polynomial.chebyshev import Chebyshev
from scipy import special
from scipy.interpolate import BPoly
from scipy.special import voigt_profile
from scipy.stats import norm


# Shape definitions
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
    logNL = nL * np.log(nL / abs_aL) - 0.5 * abs_aL**2
    tL = nL / abs_aL - abs_aL - z[mask_L]
    tL = np.maximum(tL, 1e-8)
    result[mask_L] = np.exp(logNL - nL * np.log(tL))

    # right tail
    mask_R = z >= abs_aR
    logNR = nR * np.log(nR / abs_aR) - 0.5 * abs_aR**2
    tR = nR / abs_aR - abs_aR + z[mask_R]
    tR = np.maximum(tR, 1e-8)
    result[mask_R] = np.exp(logNR - nR * np.log(tR))

    # final normalization
    norm = np.trapezoid(result, x)
    if norm <= 0 or not np.isfinite(norm):
        norm = 1e-8
    return result / norm


def double_crystal_ball_cdf(x, mu, sigma, alphaL, nL, alphaR, nR):
    betaL = alphaL
    mL = nL
    scaleL = sigma
    betaR = alphaR
    mR = nR
    scaleR = sigma
    loc = mu

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
            a_br * (b_br + beta_right) ** -m1_br / m1_br + sqrt_half * sqrt_pi * (special.erf(beta_right * sqrt_half) - special.erf(0.0))
        ) * scale_right
        r = np.empty_like(x)

        for i in range(len(x)):
            scale = T(1) / (scale_left if x[i] < loc else scale_right)
            z = (x[i] - loc) * scale
            if z < -beta_left:
                r[i] = a_bl * (b_bl - z) ** -m1_bl / m1_bl * scale_left / norm
            elif z < 0:
                r[i] = (
                    (a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(z * sqrt_half) - special.erf(-beta_left * sqrt_half)))
                    * scale_left
                    / norm
                )
            elif z < beta_right:
                r[i] = (
                    a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(0.0) - special.erf(-beta_left * sqrt_half))
                ) * scale_left + sqrt_half * sqrt_pi * (special.erf(z * sqrt_half) - special.erf(0.0)) * scale_right
                r[i] /= norm
            else:
                r[i] = (
                    a_bl * (b_bl + beta_left) ** -m1_bl / m1_bl + sqrt_half * sqrt_pi * (special.erf(0.0) - special.erf(-beta_left * sqrt_half))
                ) * scale_left + (
                    sqrt_half * sqrt_pi * (special.erf(beta_right * sqrt_half) - special.erf(0.0))
                    + a_br * (b_br + beta_right) ** -m1_br / m1_br
                    - a_br * (b_br + z) ** -m1_br / m1_br
                ) * scale_right
                r[i] /= norm
        return r

    cb1 = cdf(x, betaL, mL, scaleL, betaR, mR, scaleR, loc)

    return cb1


def double_voigtian(x, mu, sigma1, gamma1, sigma2, gamma2):
    result = voigt_profile(x - mu, sigma1, gamma1) + voigt_profile(x - mu, sigma2, gamma2)
    # Normalize
    return result / np.trapezoid(result, x)


def gaussian_pdf(x, mu, sigma):
    # normalized Gaussian
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def gaussian_cdf(x, mu, sigma):
    return 1 / 2 * (1 + special.erf((x - mu) / (np.sqrt(2) * sigma)))


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

        result[mask_core] = np.exp(-0.5 * z[mask_core] ** 2)

        # Tail region (Power law)
        # Calculate N safely using log sum exp trick
        try:
            logN = n * np.log(n / abs_alpha) - 0.5 * abs_alpha**2
            N = np.exp(logN)
        except FloatingPointError:
            N = 1e300  # fallback large number

        base = (n / abs_alpha - abs_alpha - z[mask_tail]) if (alpha < 0) else (n / abs_alpha - abs_alpha + z[mask_tail])
        base = np.clip(base, 1e-15, np.inf)  # prevent zero or negative values

        result[mask_tail] = N * base ** (-n)
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


def phase_space(x, a, b, x_min, x_max):
    # Clip exponents into a safe range
    a_clamped = np.clip(a, 0, 20)
    b_clamped = np.clip(b, 0, 20)

    # 2) Work in log - space
    t1 = np.clip(x - x_min, 1e-8, None)
    t2 = np.clip(x_max - x, 1e-8, None)

    log_pdf = a_clamped * np.log(t1) + b_clamped * np.log(t2)
    pdf = np.exp(log_pdf - np.max(log_pdf))  # subtract max for stability

    # zero outside
    pdf[(x <= x_min) | (x >= x_max)] = 0

    # Normalize
    norm = np.trapezoid(pdf, x)
    return pdf / (norm if norm > 0 else 1e-8)


def linear_pdf(x, b, C, x_min, x_max):
    x_mid = 0.5 * (x_min + x_max)
    lin = b * (x - x_mid) + C

    # Clip negative values
    lin = np.clip(lin, 0, None)

    denom = np.trapezoid(lin, x)

    return lin / denom


def linear_cdf(x, b, C, x_min, x_max):
    x = np.asarray(x)
    den = 0.5 * b * (x_max**2 - x_min**2) + C * (x_max - x_min)
    if den <= 0:
        den = 1e-8
    cdf = np.zeros_like(x, dtype=float)
    mask = (x > x_min) & (x < x_max)
    if np.any(mask):
        xm = x[mask]
        num = 0.5 * b * (xm**2 - x_min**2) + C * (xm - x_min)
        cdf[mask] = num / den
    cdf[x >= x_max] = 1.0
    return cdf


def exponential_pdf(x, C):
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


def exponential_cdf(x, C):
    cdf = 1 - np.exp(-C * x)
    return cdf


def chebyshev_background(x, *coeffs, x_min, x_max):
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)


def bernstein_poly(x, *coeffs, x_min, x_max):
    c = np.array(coeffs).reshape(-1, 1)
    return BPoly(c, [x_min, x_max])(x)


def cms(x, beta, gamma, loc):
    y = cmsshape.pdf(x, beta, gamma, loc)
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Shape parameters
def shape_params(mass, x_min, x_max):
    if mass == "Z" or mass == "Z_muon":
        SIGNAL_MODELS = {
            "dcb": {
                "pdf": double_crystal_ball_pdf,
                "cdf": double_crystal_ball_cdf,
                "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
                "bounds": {"mu": (88, 90.5, 92), "sigma": (1, 3, 6), "alphaL": (0, 1.0, 10), "nL": (0, 5.0, 30), "alphaR": (0, 1.0, 10), "nR": (0, 5.0, 30)},
            },
            "dv": {
                "pdf": double_voigtian,
                "cdf": None,
                "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
                "bounds": {"mu": (88, 90, 93), "sigma1": (2.0, 3.0, 4.0), "gamma1": (0.01, 0.5, 3.0), "sigma2": (1.0, 2.0, 3.0), "gamma2": (0.01, 1.0, 3.0)},
            },
            "g": {"pdf": gaussian_pdf, "cdf": gaussian_cdf, "params": ["mu", "sigma"], "bounds": {"mu": (88, 90, 94), "sigma": (1, 2.5, 6)}},
            "cbg": {
                "pdf": CB_G,
                "cdf": None,
                "params": ["mu", "sigma", "alpha", "n", "sigma2"],
                "bounds": {"mu": (88, 90, 92), "sigma": (1, 3, 6), "alpha": (-10, -1, 10), "n": (0, 5.0, 30), "sigma2": (1, 3, 10)},
            },
        }

        BACKGROUND_MODELS = {
            "ps": {
                "pdf": partial(phase_space, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["a", "b"],
                "bounds": {"a": (0, 0.5, 10), "b": (0, 1, 30)},
            },
            "lin": {
                "pdf": partial(linear_pdf, x_min=x_min, x_max=x_max),
                "cdf": partial(linear_cdf, x_min=x_min, x_max=x_max),
                "params": ["b", "C"],
                "bounds": {"b": (-1, 0.1, 1), "C": (0, 0.1, 10)},
            },
            "exp": {"pdf": exponential_pdf, "cdf": exponential_cdf, "params": ["C"], "bounds": {"C": (-10, 0.1, 10)}},
            "cheb": {
                "pdf": partial(chebyshev_background, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {"c0": (0.001, 1, 3), "c1": (0.001, 1, 3), "c2": (0.001, 1, 3)},
            },
            "bpoly": {
                "pdf": partial(bernstein_poly, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0, 0.05, 10),
                    "c1": (0, 0.1, 1),
                    "c2": (0, 0.1, 1),
                },
            },
            "cms": {
                "pdf": cms,
                "cdf": None,
                "params": ["beta", "gamma", "loc"],
                "bounds": {"beta": (-0.5, 0.02, 0.3), "gamma": (0, 0.074, 0.1), "loc": (50, 90, 150)},
            },
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
                    "nR": (0, 5.0, 30),
                },
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
                    "gamma2": (0.01, 1.0, 3.0),
                },
            },
            "g": {"pdf": gaussian_pdf, "cdf": gaussian_cdf, "params": ["mu", "sigma"], "bounds": {"mu": (2.5, 3.05, 3.5), "sigma": (0, 0.03, 0.1)}},
            "cbg": {
                "pdf": CB_G,
                "cdf": None,
                "params": ["mu", "sigma", "alpha", "n", "sigma2"],
                "bounds": {"mu": (2.5, 3.05, 3.5), "sigma": (0, 0.03, 0.06), "alpha": (-10, -1, 10), "n": (0.1, 5.0, 30), "sigma2": (0, 0.03, 0.15)},
            },
        }

        BACKGROUND_MODELS = {
            "ps": {
                "pdf": partial(phase_space, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["a", "b"],
                "bounds": {"a": (0, 0.5, 10), "b": (0, 1, 30)},
            },
            "lin": {
                "pdf": partial(linear_pdf, x_min=x_min, x_max=x_max),
                "cdf": partial(linear_cdf, x_min=x_min, x_max=x_max),
                "params": ["b", "C"],
                "bounds": {"b": (-1, 0.1, 1), "C": (0, 0.1, 10)},
            },
            "exp": {"pdf": exponential_pdf, "cdf": exponential_cdf, "params": ["C"], "bounds": {"C": (-10, 0.1, 10)}},
            "cheb": {
                "pdf": partial(chebyshev_background, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {"c0": (0.001, 1, 3), "c1": (0.001, 1, 3), "c2": (0.001, 1, 3)},
            },
            "bpoly": {
                "pdf": partial(bernstein_poly, x_min=x_min, x_max=x_max),
                "cdf": None,
                "params": ["c0", "c1", "c2"],
                "bounds": {
                    "c0": (0, 0.05, 10),
                    "c1": (0, 0.1, 1),
                    "c2": (0, 0.1, 1),
                },
            },
            "cms": {
                "pdf": cms,
                "cdf": None,
                "params": ["beta", "gamma", "loc"],
                "bounds": {"beta": (-0.5, 0.02, 0.3), "gamma": (0, 0.074, 0.1), "loc": (-4, 3, 10)},
            },
        }

    return SIGNAL_MODELS, BACKGROUND_MODELS
