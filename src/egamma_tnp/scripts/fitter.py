"""CLI for running fits on individual histogram bins using the new fitter framework.

Usage:
    egamma-fitter --config config.yaml --input data.root --bin 01pass
    egamma-fitter --config config.yaml --input data.root --bin 01fail --interactive
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import uproot
import yaml

# Import models to ensure PDFs are registered
import egamma_tnp.fitter.models  # noqa: F401
from egamma_tnp.fitter.basepdf import get_pdf_class
from egamma_tnp.fitter.fit import Fitter


def load_custom_models(models_file: str | Path) -> None:
    """Load custom PDF models from a Python file.

    The file should define PDF classes decorated with @register_pdf that inherit from BasePDF.

    Args:
        models_file: Path to Python file containing custom PDF models
    """
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


def parse_bin_name(bin_name: str) -> tuple[str, str]:
    """Parse bin name like '01pass' or '01fail' into bin number and pass/fail status.

    Args:
        bin_name: Bin name like '01pass' or '01fail'

    Returns:
        Tuple of (bin_number, pass_fail) like ('01', 'Pass') or ('01', 'Fail')
    """
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


def construct_histogram_name(bin_info: str, pass_fail: str) -> str:
    """Construct histogram name from bin info and pass/fail status.

    Expected histogram names are like: bin01_el_sc_eta_m2p00Tom1p57_el_pt_10p00To20p00_Pass

    Args:
        bin_info: The bin number like '01'
        pass_fail: Either 'Pass' or 'Fail'

    Returns:
        Pattern to match histogram name
    """
    return f"bin{bin_info}_*_{pass_fail}"


def fit_single_histogram(hist_name: str, h, config: dict, interactive: bool = False) -> None:
    """Fit a single histogram and print results."""
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


def main():
    parser = argparse.ArgumentParser(description="Fit individual histogram bins")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument("--bin", help="Bin to fit (e.g., '01pass' or '01fail'). If not specified, fits all bins.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive fitting")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load custom models if specified
    if "custom_models" in config:
        custom_models_path = config["custom_models"]
        # Resolve path relative to config file if it's a relative path
        if not Path(custom_models_path).is_absolute():
            custom_models_path = config_path.parent / custom_models_path
        load_custom_models(custom_models_path)

    # Open ROOT file
    with uproot.open(args.input) as f:
        all_keys = f.keys()

        if args.bin:
            # Fit single bin
            bin_num, pass_fail = parse_bin_name(args.bin)

            # Create regex pattern: bin{num}_*_{Pass/Fail}
            pattern = re.compile(rf"^bin{bin_num}_.*_{pass_fail}(?:;.*)?$")

            # Find histogram matching the pattern
            matching_keys = [key for key in all_keys if pattern.match(key)]

            if not matching_keys:
                raise ValueError(f"No histogram found matching pattern: bin{bin_num}_.*_{pass_fail}")
            if len(matching_keys) > 1:
                # Strip cycle numbers for display
                display_keys = [key.split(";")[0] for key in matching_keys]
                raise ValueError(f"Multiple histograms match pattern bin{bin_num}_.*_{pass_fail}:\n" + "\n".join(f"  - {key}" for key in display_keys))

            hist_name = matching_keys[0]
            hist_name_display = hist_name.split(";")[0]

            # Load histogram
            h = f[hist_name].to_hist()

            # Fit the histogram
            fit_single_histogram(hist_name_display, h, config, args.interactive)

        else:
            # Fit all bins
            # Pattern to match bin histograms: bin<number>_..._Pass or bin<number>_..._Fail
            bin_pattern = re.compile(r"^bin\d+_.*_(Pass|Fail)(?:;.*)?$")
            histogram_keys = [key for key in all_keys if bin_pattern.match(key)]

            if not histogram_keys:
                raise ValueError("No bin histograms found in the ROOT file")

            print(f"Found {len(histogram_keys)} histograms to fit\n")

            for hist_key in histogram_keys:
                hist_name_display = hist_key.split(";")[0]

                try:
                    # Load histogram
                    h = f[hist_key].to_hist()

                    # Fit the histogram
                    fit_single_histogram(hist_name_display, h, config, args.interactive)

                except Exception as e:
                    print(f"\nWARNING: Failed to fit {hist_name_display}")
                    print(f"Error: {e}\n")
                    continue


if __name__ == "__main__":
    main()
