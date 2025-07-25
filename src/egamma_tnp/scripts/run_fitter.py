import argparse
import json
from pathlib import Path
import uproot

from egamma_tnp.utils import fitter_sh
from egamma_tnp.utils.fitter_sh import fit_function, load_histogram, plot_combined_fit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Extract config values
    mass = config["mass"]
    root_files_DATA = config["input"]["root_files_DATA"]
    root_files_MC = config["input"]["root_files_MC"]

    fit_type = config["fit"]["fit_type"]
    use_cdf = config["fit"].get("use_cdf", False)
    sigmoid_eff = config["fit"].get("sigmoid_eff", False)
    interactive = config["fit"].get("interactive", False)
    args_bin = config["fit"].get("bin", "bin0")
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)

    print(f"x_min: {x_min}, x_max: {x_max}")

    # Get histogram names from config or use defaults
    hist_pass_name = config["fit"].get("hist_pass_name")
    hist_fail_name = config["fit"].get("hist_fail_name")
    
    # If histogram names aren't specified, try to construct them
    if not hist_pass_name or not hist_fail_name:
        bin_name = config["fit"].get("bin", "bin0")  # default to bin0
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

    if root_files_DATA:
        for root_file_path in root_files_DATA:
            print(f"\nProcessing file: {root_file_path}")

            with uproot.open(root_file_path) as f:
                hist_pass = load_histogram(f, hist_pass_name, "DATA")
                hist_fail = load_histogram(f, hist_fail_name, "DATA")

            if not hist_pass or not hist_fail:
                print(f"Warning: Failed to load histograms from {root_file_path}")
                continue

            print("Fitting...")

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
                x_max=x_max
            )
            
            all_results.append(results)

            # Create plot for each file
            plot_path = Path(config["output"]["plot_dir"])/"DATA"/args_bin
            plot_path.mkdir(parents=True, exist_ok=True)
            
            plot_combined_fit(
                results,
                plot_dir=plot_path,
                data_type="DATA",
                sigmoid_eff=sigmoid_eff,
                args_mass=mass
            )

        # Save combined results if specified
        if config["output"].get("results_file"):
            summary_path = Path(config["output"]["results_file"])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                # Combine results from all files
                combined_summary = "\n\n".join(
                    res.get("summary", "No summary available") 
                    for res in all_results
                )
                f.write(combined_summary)
            print(f"\nSaved combined fit summary to: {summary_path}")

    if root_files_MC:
        for root_file_path in root_files_MC:
            print(f"\nProcessing file: {root_file_path}")

            with uproot.open(root_file_path) as f:
                hist_pass = load_histogram(f, hist_pass_name, "MC")
                hist_fail = load_histogram(f, hist_fail_name, "MC")

            if not hist_pass or not hist_fail:
                print(f"Warning: Failed to load histograms from {root_file_path}")
                continue

            print("Fitting...")

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
                x_max=x_max
            )
            
            all_results.append(results)

            # Create plot for each file
            plot_path = Path(config["output"]["plot_dir"])/"MC"/args_bin
            plot_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to: {plot_path}")
            print(f"test: {str(plot_path.parent)}")
            
            plot_combined_fit(
                results,
                plot_dir=plot_path,
                data_type="MC",
                sigmoid_eff=sigmoid_eff,
                args_mass=mass
            )

        # Save combined results if specified
        if config["output"].get("results_file"):
            summary_path = Path(config["output"]["results_file"])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                # Combine results from all files
                combined_summary = "\n\n".join(
                    res.get("summary", "No summary available") 
                    for res in all_results
                )
                f.write(combined_summary)
            print(f"\nSaved combined fit summary to: {summary_path}")

    if not root_files_DATA and not root_files_MC:
        print("No input files specified.")

if __name__ == "__main__":
    main()
