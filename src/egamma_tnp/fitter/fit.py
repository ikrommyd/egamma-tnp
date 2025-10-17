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
        # Always use sig_frac and bkg_frac as yield parameters
        self._yield_params = ["sig_frac", "bkg_frac"]
        self._all_params = self._sig_params + self._bkg_params + self._yield_params

        def model_pdf(xe, *args):
            params = dict(zip(self._all_params, args))

            sig_kwargs = {k: params[k] for k in self._sig_params}
            bkg_kwargs = {k: params[k] for k in self._bkg_params}

            sig_pdf = self._signal_pdf.pdf(xe, **sig_kwargs)
            bkg_pdf = self._background_pdf.pdf(xe, **bkg_kwargs)

            return params["sig_frac"] * sig_pdf + params["bkg_frac"] * bkg_pdf

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
                x_points = np.linspace(a, b, 51)
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

        # Calculate total sum of variances for yield normalization
        total_variance = hist_sliced.variances().sum()

        # Handle yield parameters with default values and fraction-based initialization
        yield_defaults = {
            "sig_frac": {"init": 0.5, "limits": (0, 1.2)},
            "bkg_frac": {"init": 0.5, "limits": (0, 1.2)},
        }

        # Get yield config or use empty dict if not provided
        yields_config = config.get("yields", {})

        for param_name in ["sig_frac", "bkg_frac"]:
            # Get user config or default
            param_config = yields_config.get(param_name, yield_defaults[param_name])

            # Convert fraction to actual value
            init_fraction = param_config.get("init", yield_defaults[param_name]["init"])
            init_values[param_name] = init_fraction * total_variance

            # Convert limit fractions to actual values
            if "limits" in param_config:
                limit_fractions = param_config["limits"]
                limits[param_name] = (
                    limit_fractions[0] * total_variance,
                    limit_fractions[1] * total_variance,
                )
            else:
                default_limits = yield_defaults[param_name]["limits"]
                limits[param_name] = (
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
