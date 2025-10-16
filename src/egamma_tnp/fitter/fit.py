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
        self._yield_params = list(config["yields"].keys())
        self._all_params = self._sig_params + self._bkg_params + self._yield_params

        def model_pdf(xe, *args):
            params = dict(zip(self._all_params, args))

            sig_kwargs = {k: params[k] for k in self._sig_params}
            bkg_kwargs = {k: params[k] for k in self._bkg_params}

            sig_pdf = self._signal_pdf.pdf(xe, **sig_kwargs)
            bkg_pdf = self._background_pdf.pdf(xe, **bkg_kwargs)

            return params["n_sig"] * sig_pdf + params["n_bkg"] * bkg_pdf

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

        for param_name, param_config in config["yields"].items():
            init_values[param_name] = param_config["init"]
            if "limits" in param_config:
                limits[param_name] = param_config["limits"]

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
