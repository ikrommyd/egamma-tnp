from __future__ import annotations

import argparse
import re
from pathlib import Path

import uproot
import yaml

# Import models to ensure PDFs are registered
import egamma_tnp.fitter.models  # noqa: F401
from egamma_tnp.fitter.utils import fit_single_histogram, load_custom_models


def main():
    parser = argparse.ArgumentParser(description="Fit individual histogram bins")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument("--bin", help="Bin to fit (e.g., '1_pass' or '5_fail'). If not specified, fits all bins.")
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
            # Parse bin format: "1_pass" or "5_fail"
            if "_" not in args.bin:
                raise ValueError(f"Invalid bin format: {args.bin}. Expected format: '1_pass' or '5_fail'")

            bin_num_str, pass_fail_str = args.bin.split("_", 1)

            # Normalize pass/fail
            pass_fail = "Pass" if pass_fail_str.lower() in ["pass", "p"] else "Fail"

            # Determine the zero-padding width from the ROOT file
            # Find all bin numbers in the file
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
