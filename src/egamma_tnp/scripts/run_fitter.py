###################################################################################
#                            CONFIG.JSON FORMAT
###################################################################################
#
#   {
#     "info_level": "INFO"       < ----- Determines the output level in the terminal. "INFO",outputs useful info. "", won't display anything. "DEBUG" is for debugging
#     "mass": "Z",               < ----- Determines what mass you are fitting (Z, Z_muon, JPsi, JPsi_muon)
#     "input": {
#       "root_files_DATA": [
#           "NAME DATA 1":   ".root DATA file path 1 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME DATA 2":   ".root DATA file path 2 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME DATA 3":   ".root DATA file path 3 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#       ],
#       "root_files_MC": [
#           "NAME MC 1":     ".root MC file path 1 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME MC 2":     ".root MC file path 2 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME MC 3":     ".root MC file path 3 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#       ]
#     },
#     "fit": {
#       "fit_type": "dcb_cms"    < ----- Format is: (signal shape)_(background shape). Signal shapes: (dcb, g, dv, cbg), Background shapes: (lin, exp, cms, bpoly, cheb, ps)
#       "use_cdf": false,        < ----- If a shape does not have a cdf version, defaults back to pdf
#       "sigmoid_eff": false,    < ----- Switches to an unbounded efficiency that is transformed back between 0 and 1
#       "bin": "bin(number)",    < ----- Specify which pT range you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
#       "interactive": true,     < ----- Turns on interactive window for fitting (very useful for difficult fits)
#       "x_min": 70,             < ----- x range minimum for plotting
#       "x_max": 110,            < ----- x range maximum for plotting
#       "abseta": 1,             < ----- Only impacts muon .root files***. Defines absolute eta ranges
#       "numerator": "gold",     < ----- Only impacts muon .root files***. Defines numerator for efficiencies
#       "denominator": "blp"     < ----- Only impacts muon .root files***. Defines denominator for efficiencies
#     },
#     "output": {
#       "plot_dir": "",          < ----- Sets location to save plots to (if left blank, it won't save)
#       "results_file": ""       < ----- Sets location to save results to (if left blank, it won't save)
#     }
#   }
###################################################################################
# RUN EXAMPLE CONFIG FILE: run-fitter --config src/egamma_tnp/config/config_example.json
###################################################################################
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uproot

from egamma_tnp.utils import fitter_sh
from egamma_tnp.utils.fit_function import fit_function, logging
from egamma_tnp.utils.fitter_plot_model import load_histogram, plot_combined_fit


def main():
    # Reads in config file specified in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Extract config values
    mass = config["mass"]
    root_files_DATA = config["input"].get("root_files_DATA", {})
    root_files_MC = config["input"].get("root_files_MC", {})

    # Get fit parameters from config
    fit_type = config["fit"]["fit_type"]
    use_cdf = config["fit"].get("use_cdf", False)
    sigmoid_eff = config["fit"].get("sigmoid_eff", False)
    interactive = config["fit"].get("interactive", False)
    args_bin = config["fit"].get("bin", "bin0")
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)

    # Get histogram names from config or use defaults
    hist_pass_name = config["fit"].get("hist_pass_name")
    hist_fail_name = config["fit"].get("hist_fail_name")

    # If histogram names aren't specified, try to construct them
    if not hist_pass_name or not hist_fail_name:
        bin_name = config["fit"].get("bin", "bin1")

        # num and den are only necessary for Muon fits
        num = config["fit"].get("numerator", "gold")
        den = config["fit"].get("denominator", "baselineplus")

        if mass in ["Z", "JPsi"]:
            # For electron TnP
            bin_suffix, _ = fitter_sh.BINS_INFO[bin_name]
            hist_pass_name = f"{bin_name}_{bin_suffix}_Pass"
            hist_fail_name = f"{bin_name}_{bin_suffix}_Fail"
        elif mass in ["Z_muon", "JPsi_muon"]:
            # For muon TnP
            abseta = config["fit"].get("abseta", 1)
            pt = int(bin_name.split("bin")[1])
            hist_pass_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt}_Pass"
            hist_fail_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt}_Fail"
        else:
            raise ValueError(f"Unknown mass: {mass}")

    all_results = []

    pass_fail_summary = []

    if root_files_DATA:
        for plot_name, root_file_path in root_files_DATA.items():
            logging.info(f"\nProcessing file: {root_file_path}\n")

            # Load histograms from ROOT file
            with uproot.open(root_file_path) as f:
                hist_pass = load_histogram(f, hist_pass_name, "DATA")
                hist_fail = load_histogram(f, hist_fail_name, "DATA")

            if not hist_pass or not hist_fail:
                logging.warning(f"Warning: Failed to load histograms from {root_file_path}")
                continue

            # Fitting Step
            results = fit_function(
                fit_type,
                hist_pass,
                hist_fail,
                use_cdf=use_cdf,
                args_bin=args_bin,
                args_data="DATA",
                args_mass=mass,
                sigmoid_eff=sigmoid_eff,
                interactive=interactive,
                x_min=x_min,
                x_max=x_max,
            )

            all_results.append(results)

            # Create plot for each file
            plot_path = Path(config["output"]["plot_dir"]) / "DATA" / args_bin
            plot_path.mkdir(parents=True, exist_ok=True)

            # If config file has no path to save plots to, it won't save
            if config["output"].get("plot_dir"):
                fig_pass, fig_fail = plot_combined_fit(results, plot_dir=plot_path, data_type="DATA", sigmoid_eff=sigmoid_eff, args_mass=mass)
                fig_pass.savefig(plot_path / f"{plot_name}_Pass.png", bbox_inches="tight", dpi=300)
                fig_fail.savefig(plot_path / f"{plot_name}_Fail.png", bbox_inches="tight", dpi=300)

            if results["message"].is_valid:
                pass_fail_summary.append(f"{plot_name} fit passed!")
            else:
                pass_fail_summary.append(f"{plot_name} fit failed!")

    if root_files_MC:
        for plot_name, root_file_path in root_files_MC.items():
            logging.info(f"\nProcessing file: {root_file_path}\n")

            with uproot.open(root_file_path) as f:
                hist_pass = load_histogram(f, hist_pass_name, "MC")
                hist_fail = load_histogram(f, hist_fail_name, "MC")

            if not hist_pass or not hist_fail:
                logging.warning(f"Warning: Failed to load histograms from {root_file_path}")
                continue

            results = fit_function(
                fit_type,
                hist_pass,
                hist_fail,
                use_cdf=use_cdf,
                args_bin=args_bin,
                args_data="MC",
                args_mass=mass,
                sigmoid_eff=sigmoid_eff,
                interactive=interactive,
                x_min=x_min,
                x_max=x_max,
            )

            all_results.append(results)

            # Create plot for each file
            plot_path = Path(config["output"]["plot_dir"]) / "MC" / args_bin
            plot_path.mkdir(parents=True, exist_ok=True)

            if config["output"].get("plot_dir"):
                fig_pass, fig_fail = plot_combined_fit(results, plot_dir=plot_path, data_type="MC", sigmoid_eff=sigmoid_eff, args_mass=mass)
                fig_pass.savefig(plot_path / f"{plot_name}_Pass.png", bbox_inches="tight", dpi=300)
                fig_fail.savefig(plot_path / f"{plot_name}_Fail.png", bbox_inches="tight", dpi=300)

            if results["message"].is_valid:
                pass_fail_summary.append(f"{plot_name} fit passed!")
            else:
                pass_fail_summary.append(f"{plot_name} fit failed!")

        # Save combined results if specified
        if config["output"].get("results_file"):
            summary_path = Path(config["output"]["results_file"])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                # Combine results from all files
                combined_summary = "\n\n".join(res.get("summary", "No summary available") for res in all_results)
                f.write(combined_summary)
            logging.info(f"\nSaved combined fit summary to: {summary_path}")

    if not root_files_DATA and not root_files_MC:
        logging.warning("No input files specified.")

    logging.info("\nFit Summary:")
    logging.info("\n".join(pass_fail_summary))


if __name__ == "__main__":
    main()
