from __future__ import annotations

from inspect import Parameter, Signature

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL


class Fitter:
    def __init__(self, hist, signal_pdf, background_pdf, fit_range, config):
        self._hist = hist
        self._signal_pdf = signal_pdf
        self._background_pdf = background_pdf
        self._fit_range = fit_range
        self._config = config

        hist_sliced = hist[fit_range[0] * 1j : fit_range[1] * 1j]

        self._bin_edges = hist_sliced.axes[0].edges
        self._bin_values = hist_sliced.values()

        self._sig_params = list(config["signal"].keys())
        self._bkg_params = list(config["background"].keys())
        # Use only n_sig; n_bkg is computed as total_variance - n_sig
        self._yield_params = ["n_sig"]
        self._all_params = self._sig_params + self._bkg_params + self._yield_params

        # Calculate total variance first (needed for model_pdf closure)
        hist_sliced_temp = hist[fit_range[0] * 1j : fit_range[1] * 1j]
        total_variance = hist_sliced_temp.variances().sum()

        def model_pdf(xe, *args):
            params = dict(zip(self._all_params, args))

            sig_kwargs = {k: params[k] for k in self._sig_params}
            bkg_kwargs = {k: params[k] for k in self._bkg_params}

            sig_pdf = self._signal_pdf.pdf(xe, **sig_kwargs)
            bkg_pdf = self._background_pdf.pdf(xe, **bkg_kwargs)

            # n_sig is the signal count; n_bkg = total_variance - n_sig
            n_sig = params["n_sig"]
            n_bkg = total_variance - n_sig
            return n_sig * sig_pdf + n_bkg * bkg_pdf

        # Tell iminuit the parameter names using __signature__
        params_signature = [Parameter("xe", Parameter.POSITIONAL_OR_KEYWORD)]
        params_signature.extend([Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in self._all_params])
        model_pdf.__signature__ = Signature(params_signature)

        cost = ExtendedBinnedNLL(self._bin_values, self._bin_edges, model_pdf, use_pdf="approximate")

        def _pred_simpson(args):
            from scipy.integrate import simpson

            d = np.empty(len(self._bin_edges) - 1)
            for i in range(len(self._bin_edges) - 1):
                a = self._bin_edges[i]
                b = self._bin_edges[i + 1]
                x_points = np.linspace(a, b, 11)
                y_points = model_pdf(x_points, *args)
                d[i] = simpson(y_points, x=x_points)
            return d

        # cost._pred_impl = _pred_simpson

        init_values = {}
        limits = {}

        for param_name, param_config in config["signal"].items():
            init_values[param_name] = param_config["init"]
            if "limits" in param_config:
                limits[param_name] = param_config["limits"]

        for param_name, param_config in config["background"].items():
            init_values[param_name] = param_config["init"]
            if "limits" in param_config:
                limits[param_name] = param_config["limits"]

        # Handle yield parameters with default values and fraction-based initialization
        # User specifies sig_frac (0-1), which gets converted to n_sig (actual count)
        yield_defaults = {
            "sig_frac": {"init": 0.5, "limits": (0, 1)},
        }

        # Get yield config or use empty dict if not provided
        yields_config = config.get("yields", {})

        # Handle n_sig parameter (signal count)
        sig_frac_config = yields_config.get("sig_frac", yield_defaults["sig_frac"])

        # Convert sig_frac fraction to actual n_sig count
        init_sig_frac = sig_frac_config.get("init", yield_defaults["sig_frac"]["init"])
        init_values["n_sig"] = init_sig_frac * total_variance

        # Convert limit fractions to actual values
        if "limits" in sig_frac_config:
            limit_fractions = sig_frac_config["limits"]
            limits["n_sig"] = (
                limit_fractions[0] * total_variance,
                limit_fractions[1] * total_variance,
            )
        else:
            default_limits = yield_defaults["sig_frac"]["limits"]
            limits["n_sig"] = (
                default_limits[0] * total_variance,
                default_limits[1] * total_variance,
            )

        self.minuit = Minuit(cost, **init_values)
        self.minuit.strategy = 2

        for param_name, param_limits in limits.items():
            self.minuit.limits[param_name] = param_limits

    def fit(self):
        self.minuit.migrad()
        return self.minuit

    def interactive(self):
        self.minuit.interactive()
        return self.minuit
