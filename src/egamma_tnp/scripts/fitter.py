"""CLI for running fits on individual histogram bins using the new fitter framework.

Usage:
    egamma-fitter --config config.yaml --input data.root --bin 01pass
    egamma-fitter --config config.yaml --input data.root --bin 01fail --interactive
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import uproot
import yaml

# Import models to ensure PDFs are registered
import egamma_tnp.fitter.models  # noqa: F401
from egamma_tnp.fitter.basepdf import get_pdf_class
from egamma_tnp.fitter.fit import Fitter


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


def main():
    parser = argparse.ArgumentParser(description="Fit individual histogram bins")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument("--bin", required=True, help="Bin to fit (e.g., '01pass' or '01fail')")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive fitting")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Parse bin name
    bin_num, pass_fail = parse_bin_name(args.bin)

    # Open ROOT file and find matching histogram
    with uproot.open(args.input) as f:
        # Create regex pattern: bin{num}_*_{Pass/Fail}
        pattern = re.compile(rf"^bin{bin_num}_.*_{pass_fail}(?:;.*)?$")

        # Find histogram matching the pattern
        all_keys = f.keys()
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

    if args.interactive:
        result = fitter.interactive()
    else:
        result = fitter.fit()

    # Print minuit object
    print(f"\nHistogram: {hist_name_display}")
    print(f"Fit range: {fit_range}\n")
    print(result)


if __name__ == "__main__":
    main()
