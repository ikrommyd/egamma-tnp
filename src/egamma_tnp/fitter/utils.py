from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from egamma_tnp.fitter.basepdf import get_pdf_class
from egamma_tnp.fitter.fit import Fitter


def load_custom_models(models_file):
    models_path = Path(models_file).resolve()

    if not models_path.exists():
        raise FileNotFoundError(f"Custom models file not found: {models_path}")

    # Load the module from file
    spec = importlib.util.spec_from_file_location("custom_models", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {models_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_models"] = module
    spec.loader.exec_module(module)


def parse_bin_name(bin_name):
    # Extract bin number (all digits at start)
    bin_num = ""
    for char in bin_name:
        if char.isdigit():
            bin_num += char
        else:
            break

    # Extract pass/fail status
    remainder = bin_name[len(bin_num) :].lower()
    if "pass" in remainder:
        pass_fail = "Pass"
    elif "fail" in remainder:
        pass_fail = "Fail"
    else:
        raise ValueError(f"Could not determine pass/fail status from bin name: {bin_name}")

    return bin_num, pass_fail


def fit_single_histogram(hist_name, h, config, interactive=False):
    # Extract fit configuration
    fit_range = tuple(config["fit_range"])

    # Create PDFs from config
    signal_config = config["signal"]
    signal_pdf_name = signal_config["pdf"]
    SignalPDF = get_pdf_class(signal_pdf_name)
    sig_pdf = SignalPDF(fit_range)

    background_config = config["background"]
    background_pdf_name = background_config["pdf"]
    BackgroundPDF = get_pdf_class(background_pdf_name)
    bkg_pdf = BackgroundPDF(fit_range)

    # Build parameter config for Fitter
    param_config = {
        "signal": signal_config.get("parameters", {}),
        "background": background_config.get("parameters", {}),
    }

    # Add yields if specified
    if "yields" in config:
        param_config["yields"] = config["yields"]

    # Create and run fitter
    fitter = Fitter(h, sig_pdf, bkg_pdf, fit_range, param_config)

    if interactive:
        result = fitter.interactive()
    else:
        result = fitter.fit()

    # Print minuit object
    print(f"\nHistogram: {hist_name}")
    print(f"Fit range: {fit_range}\n")
    print(result)
