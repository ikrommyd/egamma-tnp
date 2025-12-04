from __future__ import annotations

import argparse
from pathlib import Path

import uproot
import yaml

# Import models to ensure PDFs are registered
import egamma_tnp.fitter.models  # noqa: F401
from egamma_tnp.fitter.utils import (
    fit_single_histogram,
    get_histogram_keys,
    load_custom_models,
    process_mc_histogram,
)


def main():
    parser = argparse.ArgumentParser(description="Fit individual histogram bins")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", help="Path to data ROOT file (will be fitted)")
    parser.add_argument("--mc", help="Path to MC ROOT file (will be summed, no fitting)")
    parser.add_argument("--bin", help="Bin to process (e.g., '1_pass' or '5_fail'). If not specified, processes all bins.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive fitting (data only)")
    args = parser.parse_args()

    # Require at least one of --data or --mc
    if not args.data and not args.mc:
        parser.error("At least one of --data or --mc must be specified")

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Process MC file first (sum histograms)
    if args.mc:
        print(f"Processing MC file: {args.mc}")
        print("=" * 60)
        with uproot.open(args.mc) as f:
            all_keys = f.keys()
            histogram_keys = get_histogram_keys(all_keys, args.bin)

            print(f"Found {len(histogram_keys)} histogram(s) to process\n")

            for hist_key in histogram_keys:
                hist_name_display = hist_key.split(";")[0]

                try:
                    h = f[hist_key].to_hist()
                    process_mc_histogram(hist_name_display, h, config)
                except Exception as e:
                    print(f"\nWARNING: Failed to process {hist_name_display}")
                    print(f"Error: {e}\n")
                    continue

    # Load custom models if specified (only needed for data fitting)
    if args.data and "custom_models" in config:
        custom_models_path = config["custom_models"]
        # Resolve path relative to config file if it's a relative path
        if not Path(custom_models_path).is_absolute():
            custom_models_path = config_path.parent / custom_models_path
        load_custom_models(custom_models_path)

    # Process data file (fit histograms)
    if args.data:
        print(f"Processing data file: {args.data}")
        print("=" * 60)
        with uproot.open(args.data) as f:
            all_keys = f.keys()
            histogram_keys = get_histogram_keys(all_keys, args.bin)

            print(f"Found {len(histogram_keys)} histogram(s) to fit\n")

            for hist_key in histogram_keys:
                hist_name_display = hist_key.split(";")[0]

                try:
                    h = f[hist_key].to_hist()
                    fit_single_histogram(hist_name_display, h, config, args.interactive)
                except Exception as e:
                    print(f"\nWARNING: Failed to fit {hist_name_display}")
                    print(f"Error: {e}\n")
                    continue


if __name__ == "__main__":
    main()
