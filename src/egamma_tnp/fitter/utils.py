from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

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

    # Print histogram info before fitting
    print(f"\nHistogram: {hist_name}")
    print(f"Fit range: {fit_range}\n")

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
    print(result)

    # Print signal yield
    n_sig = result.values["n_sig"]
    n_sig_err = result.errors["n_sig"]
    print(f"\nTotal Signal Yield: {n_sig:.1f} ± {n_sig_err:.1f}\n")


def process_mc_histogram(hist_name, h, config):
    """Process MC histogram by summing counts (no fitting)."""
    # Get fit range from config
    fit_range = tuple(config["fit_range"])

    # Slice histogram to fit range
    hist_sliced = h[fit_range[0] * 1j : fit_range[1] * 1j]

    # Sum the values
    total_count = hist_sliced.values().sum()
    # Uncertainty is sqrt(N) for counting statistics
    uncertainty = np.sqrt(total_count)

    print(f"\nHistogram: {hist_name}")
    print(f"Range: {fit_range}")
    print(f"Total Signal Yield: {total_count:.1f} ± {uncertainty:.1f}\n")


def get_histogram_keys(all_keys, bin_arg=None):
    """Get histogram keys to process, optionally filtered by bin argument."""
    if bin_arg:
        # Process single bin
        if "_" not in bin_arg:
            raise ValueError(f"Invalid bin format: {bin_arg}. Expected format: '1_pass' or '5_fail'")

        bin_num_str, pass_fail_str = bin_arg.split("_", 1)

        # Normalize pass/fail
        pass_fail = "Pass" if pass_fail_str.lower() in ["pass", "p"] else "Fail"

        # Determine the zero-padding width from the ROOT file
        bin_pattern_all = re.compile(r"^bin(\d+)_.*_(Pass|Fail)(?:;.*)?$")
        bin_numbers = []
        for key in all_keys:
            match = bin_pattern_all.match(key)
            if match:
                bin_numbers.append(match.group(1))

        if not bin_numbers:
            raise ValueError("No bin histograms found in the ROOT file")

        # Determine padding width from the bin numbers
        max_width = max(len(bn) for bn in bin_numbers)

        # Pad the user-provided bin number
        bin_num = bin_num_str.zfill(max_width)

        # Create regex pattern: bin{num}_*_{Pass/Fail}
        pattern = re.compile(rf"^bin{bin_num}_.*_{pass_fail}(?:;.*)?$")

        # Find histogram matching the pattern
        matching_keys = [key for key in all_keys if pattern.match(key)]

        if not matching_keys:
            raise ValueError(f"No histogram found matching pattern: bin{bin_num}_.*_{pass_fail}")
        if len(matching_keys) > 1:
            display_keys = [key.split(";")[0] for key in matching_keys]
            raise ValueError(f"Multiple histograms match pattern bin{bin_num}_.*_{pass_fail}:\n" + "\n".join(f"  - {key}" for key in display_keys))

        return [matching_keys[0]]
    else:
        # Process all bins
        bin_pattern = re.compile(r"^bin\d+_.*_(Pass|Fail)(?:;.*)?$")
        histogram_keys = [key for key in all_keys if bin_pattern.match(key)]

        if not histogram_keys:
            raise ValueError("No bin histograms found in the ROOT file")

        return histogram_keys
