from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

# PDF Registry
_PDF_REGISTRY = {}


def register_pdf(name: str):
    """Decorator to register a PDF class with a string name."""

    def decorator(cls):
        _PDF_REGISTRY[name] = cls
        return cls

    return decorator


def get_pdf_class(name: str):
    """Get a PDF class by its registered name."""
    if name not in _PDF_REGISTRY:
        raise ValueError(f"PDF '{name}' not found in registry. Available PDFs: {list(_PDF_REGISTRY.keys())}")
    return _PDF_REGISTRY[name]


class BasePDF(ABC):
    def __init__(self, norm_range):
        self.norm_range = norm_range

    @abstractmethod
    def _unnormalized_pdf(self, x, **params):
        pass

    def _unnormalized_cdf(self, x, **params):
        return None

    def _analytic_integrate(self, x_low, x_high, **params):
        return None

    def pdf(self, x, **params):
        x = np.asarray(x)
        unnorm = self._unnormalized_pdf(x, **params)
        norm = self._get_normalization(**params)
        return unnorm / norm

    def cdf(self, x, **params):
        x = np.asarray(x)
        unnorm_cdf = self._unnormalized_cdf(x, **params)
        if unnorm_cdf is not None:
            norm = self._get_normalization(**params)
            return unnorm_cdf / norm

        # Fall back to numerical integration to compute CDF
        # CDF(x) = integral from norm_range[0] to x
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]

        cdf_vals = np.zeros(len(x))
        for i, xi in enumerate(x_sorted):
            if xi <= self.norm_range[0]:
                cdf_vals[x_sorted_idx[i]] = 0.0
            elif xi >= self.norm_range[1]:
                cdf_vals[x_sorted_idx[i]] = 1.0
            else:
                integral = self._numeric_integrate(self.norm_range[0], xi, use_unnormalized=False, **params)[0]
                cdf_vals[x_sorted_idx[i]] = integral

        return cdf_vals

    def integrate(self, x_low, x_high, **params):
        x_low = np.asarray(x_low)
        x_high = np.asarray(x_high)

        # Try analytic integration first
        analytic = self._analytic_integrate(x_low, x_high, **params)
        if analytic is not None:
            norm = self._get_normalization(**params)
            return analytic / norm

        # Try CDF subtraction
        cdf_low = self._unnormalized_cdf(x_low, **params)
        cdf_high = self._unnormalized_cdf(x_high, **params)

        if cdf_low is not None and cdf_high is not None:
            norm = self._get_normalization(**params)
            return (cdf_high - cdf_low) / norm

        # Fall back to numerical integration
        return self._numeric_integrate(x_low, x_high, **params)

    def _get_normalization(self, **params):
        # Try analytic integration first
        analytic = self._analytic_integrate(
            np.array([self.norm_range[0]]),
            np.array([self.norm_range[1]]),
            **params,
        )
        if analytic is not None:
            norm = analytic[0]
            if norm > 0 and np.isfinite(norm):
                return norm
            else:
                warnings.warn(
                    f"Analytic integration normalization is invalid (norm={norm}). Falling back to CDF or numerical integration.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Try CDF subtraction
        cdf_vals = self._unnormalized_cdf(np.array(self.norm_range), **params)
        if cdf_vals is not None:
            norm = cdf_vals[1] - cdf_vals[0]
            if norm > 0 and np.isfinite(norm):
                return norm
            else:
                warnings.warn(
                    f"CDF normalization is invalid (norm={norm}). Falling back to numerical integration.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Fall back to numerical integration
        return self._numeric_integrate(self.norm_range[0], self.norm_range[1], use_unnormalized=True, **params)[0]

    def _numeric_integrate(self, x_low, x_high, n_points=1001, method="simpson", use_unnormalized=False, **params):
        x_low = np.atleast_1d(x_low)
        x_high = np.atleast_1d(x_high)

        pdf_func = self._unnormalized_pdf if use_unnormalized else self.pdf

        if method == "midpoint":
            x_mid = 0.5 * (x_low + x_high)
            widths = x_high - x_low
            heights = pdf_func(x_mid, **params)
            return widths * heights

        elif method == "trapezoid":
            integrals = np.zeros(len(x_low))
            for i, (low, high) in enumerate(zip(x_low, x_high)):
                x_points = np.linspace(low, high, n_points)
                y_points = pdf_func(x_points, **params)
                integrals[i] = np.trapezoid(y_points, x_points)
            return integrals

        elif method == "simpson":
            from scipy.integrate import simpson

            integrals = np.zeros(len(x_low))
            for i, (low, high) in enumerate(zip(x_low, x_high)):
                n_pts = n_points if n_points % 2 == 1 else n_points + 1
                x_points = np.linspace(low, high, n_pts)
                y_points = pdf_func(x_points, **params)
                integrals[i] = simpson(y_points, x=x_points)
            return integrals

        elif method == "quad":
            from scipy.integrate import quad

            integrals = np.zeros(len(x_low))
            for i, (low, high) in enumerate(zip(x_low, x_high)):
                result, _ = quad(lambda x: pdf_func(x, **params), low, high, limit=50)
                integrals[i] = result
            return integrals

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def __call__(self, x, **params):
        return self.pdf(x, **params)
