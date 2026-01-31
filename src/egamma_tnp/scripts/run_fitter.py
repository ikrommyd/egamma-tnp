###################################################################################
#                            CONFIG.JSON FORMAT
###################################################################################
#
#   {
#     "info_level": "INFO"       < ----- INFO or DEBUG (more verbose)
#     "mass": "Z",                 < ----- Determines what mass you are fitting (Z, Z_muon, JPsi, JPsi_muon)
#     "input": {
#       "root_files_DATA": [                                  < ----- The name will be the name of the plot file that is saved in plot_dir
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
#       "bin_ranges": [[5,7], [7,10], [10,20], [20,45], [45,75], [75,500]],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
#       "bin": ["bin0", "bin1, etc"],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
#       "fit_type": "dcb_cms"    < ----- Format is: (signal shape)_(background shape). Signal shapes: (dcb, g, dv, cbg), Background shapes: (lin, exp, cms, bpoly, cheb, ps)
#       "use_cdf": false,        < ----- If a shape does not have a cdf version, defaults back to pdf
#       "sigmoid_eff": false,    < ----- Switches to an unbounded efficiency that is transformed back between 0 and 1
#       "interactive": true,     < ----- Turns on interactive window for fitting (very useful for difficult fits)
#       "x_min": 70,             < ----- x range minimum for plotting
#       "x_max": 110,            < ----- x range maximum for plotting
#       "abseta": 1,             < ----- ***Only impacts muon .root files. Defines absolute eta ranges
#       "numerator": "gold",     < ----- ***Only impacts muon .root files. Defines numerator for efficiencies
#       "denominator": "blp"     < ----- ***Only impacts muon .root files. Defines denominator for efficiencies
#     },
#     "output": {
#       "plot_dir": "",          < ----- Sets location to save plots to (if left blank, it won't save)
#       "results_file": ""       < ----- Sets location to save results to (if left blank, it won't save)
#    },
#    "scale_factors": {
#        "data_mc_pair": {                                      < ----- Creates explicit scale factors for pairs of data and MC files (useful for comparing one file to multiple others)
#            "Scale Factor 1": ["NAME DATA 1", "NAME MC 1"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
#            "Scale Factor 2": ["NAME DATA 2", "NAME MC 2"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
#            "Scale Factor 3": ["NAME DATA 3", "NAME MC 3"]     < ----- Outputs scale factor of two file specified. DATA must be put before MC
#     }
#    }
#  }

###################################################################################
# RUN ON A CONFIG FILE: run-fitter --config config.json
###################################################################################
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from egamma_tnp.utils.fitter_utils import fit_function, get_bin_info, load_histogram, plot_combined_fit
from egamma_tnp.utils.logger_utils import CustomTimeElapsedColumn, print_efficiency_summary, setup_logger

console = Console()

COLOR_BORDER = "#00B4D8"
COLOR_PRIMARY = "#E6EDF3"
COLOR_SECONDARY = "#AAB3BF"
COLOR_SUCCESS = "#06D6A0"
COLOR_ERROR = "#E63946"
COLOR_HIGHLIGHT = "#FFB703"
COLOR_WARNING = "#F77F00"
COLOR_BG_DARK = "#0D1117"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    mass = config["mass"]
    root_files_DATA = config["input"].get("root_files_DATA", {})
    root_files_MC = config["input"].get("root_files_MC", {})

    fit_type = config["fit"]["fit_type"]
    use_cdf = config["fit"].get("use_cdf", False)
    sigmoid_eff = config["fit"].get("sigmoid_eff", False)
    interactive = config["fit"].get("interactive", False)
    args_bin = config["fit"].get("bin", "bin0")
    if isinstance(args_bin, str):
        args_bin = [args_bin]
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)
    info = config["info_level"]
    if info == "DEBUG":
        logger = setup_logger(level="DEBUG")
    else:
        logger = setup_logger(level="INFO")

    all_results = []
    data_msg_per_bin = []
    mc_msg_per_bin = []
    data_msg = []
    mc_msg = []
    data_eff_per_bin, data_err_per_bin, mc_eff_per_bin, mc_err_per_bin, sf_per_bin, sf_err_per_bin = [], [], [], [], [], []
    output_tables_data = defaultdict(list)
    output_progs_data = defaultdict(list)
    output_texts_data = defaultdict(list)
    output_tables_mc = defaultdict(list)
    output_progs_mc = defaultdict(list)
    output_texts_mc = defaultdict(list)
    all_pt_bins = []

    data_eff_dict = defaultdict(dict)
    data_err_dict = defaultdict(dict)
    mc_eff_dict = defaultdict(dict)
    mc_err_dict = defaultdict(dict)

    console = Console()

    bins_progress = Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),  # <-- Only here
        console=console,
    )

    # Sub progress bars (DATA/MC per bin)
    sub_progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        CustomTimeElapsedColumn(),
        console=console,
    )

    progress_group = Group(bins_progress, sub_progress)

    with Live(progress_group, refresh_per_second=5, console=console):
        total_fits = len(args_bin) * (len(root_files_DATA) + len(root_files_MC))
        task_bins = bins_progress.add_task("Fitting bins", total=total_fits)

        for pt in args_bin:
            data_eff_list = []
            data_err_list = []
            mc_eff_list = []
            mc_err_list = []
            sf_list = []
            sf_err_list = []
            data_msg_parts = []
            mc_msg_parts = []

            has_data = len(root_files_DATA) > 0
            has_mc = len(root_files_MC) > 0

            task_data = None
            task_mc = None

            if has_data:
                task_data = sub_progress.add_task(f"    DATA ({pt})", total=len(root_files_DATA))
            else:
                sub_progress.console.print(f"[yellow]No DATA files to fit for {pt}")

            if has_mc:
                task_mc = sub_progress.add_task(f"    MC   ({pt})", total=len(root_files_MC))
            else:
                sub_progress.console.print(f"[yellow]No MC files to fit for {pt}")

            hist_pass_name = config["fit"].get("hist_pass_name")
            hist_fail_name = config["fit"].get("hist_fail_name")
            BINS_INFO = get_bin_info(mass, config["fit"].get("bin_ranges", []))
            if not hist_pass_name or not hist_fail_name:
                bin_suffix, bin_range = BINS_INFO[pt]
                if mass in ["Z", "JPsi"]:
                    hist_pass_name = f"{pt}_{bin_suffix}_Pass"
                    hist_fail_name = f"{pt}_{bin_suffix}_Fail"
                else:
                    abseta = config["fit"].get("abseta", 1)
                    pt_hist = int(pt.split("bin")[1])
                    num = config["fit"].get("numerator", "gold")
                    den = config["fit"].get("denominator", "baselineplus")
                    hist_pass_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt_hist}_Pass"
                    hist_fail_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt_hist}_Fail"

            data_items = list(root_files_DATA.items())
            mc_items = list(root_files_MC.items())

            # Pad shorter list with (None, None)
            max_len = max(len(data_items), len(mc_items))
            data_items += [(None, None)] * (max_len - len(data_items))
            mc_items += [(None, None)] * (max_len - len(mc_items))

            data_eff_all = {}
            data_err_all = {}
            mc_eff_all = {}
            mc_err_all = {}

            style_data = "green"
            style_mc = "green"

            for i in range(max_len):
                data_key, data_path = data_items[i]
                mc_key, mc_path = mc_items[i]

                # Defaults
                data_eff, data_err = None, None
                mc_eff, mc_err = None, None
                data_msg = None
                mc_msg = None

                # ---- Process DATA ----
                if data_key is not None and has_data:
                    sub_progress.update(task_data, description=f"    [cyan]DATA ({pt}): [yellow]{data_key}")

                    with uproot.open(data_path) as f:
                        h_pass = load_histogram(f, hist_pass_name, "DATA")
                        h_fail = load_histogram(f, hist_fail_name, "DATA")

                    res_data = fit_function(
                        config,
                        fit_type,
                        h_pass,
                        h_fail,
                        use_cdf=use_cdf,
                        args_bin=pt,
                        args_data="DATA",
                        args_mass=mass,
                        sigmoid_eff=sigmoid_eff,
                        interactive=interactive,
                        x_min=x_min,
                        x_max=x_max,
                        data_name=data_key,
                        mc_name=None,
                    )

                    # store per-pt nested
                    data_eff_dict[pt][data_key] = res_data["popt"]["epsilon"]
                    data_err_dict[pt][data_key] = res_data["perr"]["epsilon"]

                    if data_key and data_eff is not None:
                        data_eff_all[data_key] = data_eff
                        data_err_all[data_key] = data_err

                    plot_path = Path(config["output"]["plot_dir"]) / "DATA" / pt
                    plot_path.mkdir(parents=True, exist_ok=True)

                    if config["output"].get("plot_dir"):
                        fig_pass, fig_fail = plot_combined_fit(config, res_data, plot_path, data_type="DATA", sigmoid_eff=sigmoid_eff, args_mass=mass)
                        fig_pass.savefig(plot_path / f"{data_key}_Pass.png", bbox_inches="tight", dpi=300)
                        fig_fail.savefig(plot_path / f"{data_key}_Fail.png", bbox_inches="tight", dpi=300)
                        plt.close(fig_pass)
                        plt.close(fig_fail)
                        plt.close("all")

                    output_tables_data[pt].append(res_data.get("sum_table", ""))
                    output_progs_data[pt].append(res_data.get("sum_prog", ""))
                    output_texts_data[pt].append(res_data.get("sum_text", ""))

                    all_results.append(res_data)

                    eps, err = res_data["popt"]["epsilon"], res_data["perr"]["epsilon"]
                    data_eff, data_err = eps, err
                    if data_key is not None:
                        data_msg = f"{data_key} fit {'passed' if res_data['message'].is_valid else 'failed'}"
                    else:
                        data_msg = "DATA N/A"

                    if data_msg:
                        data_msg_parts.append(data_msg)
                    else:
                        data_msg_parts.append("N/A")

                    style_data = "green" if res_data["message"].is_valid else "red"
                    sub_progress.update(task_data, advance=1, style=style_data)
                    bins_progress.update(task_bins, advance=1)  # <-- update main bin progress

                elif has_data:
                    # No data file for this index, still advance progress bar
                    sub_progress.update(task_data, advance=1, completed=i + 1)
                    bins_progress.update(task_bins, advance=1)  # <-- update main bin progress

                # ---- Process MC ----
                if mc_key is not None and has_mc:
                    sub_progress.update(task_mc, description=f"    [cyan]MC   ({pt}): [yellow]{mc_key}")

                    with uproot.open(mc_path) as f:
                        h_pass = load_histogram(f, hist_pass_name, "MC")
                        h_fail = load_histogram(f, hist_fail_name, "MC")

                    res_mc = fit_function(
                        config,
                        fit_type,
                        h_pass,
                        h_fail,
                        use_cdf=use_cdf,
                        args_bin=pt,
                        args_data="MC",
                        args_mass=mass,
                        sigmoid_eff=sigmoid_eff,
                        interactive=interactive,
                        x_min=x_min,
                        x_max=x_max,
                        data_name=None,
                        mc_name=mc_key,
                    )

                    mc_eff_dict[pt][mc_key] = res_mc["popt"]["epsilon"]
                    mc_err_dict[pt][mc_key] = res_mc["perr"]["epsilon"]

                    if mc_key and mc_eff is not None:
                        mc_eff_all[mc_key] = mc_eff
                        mc_err_all[mc_key] = mc_err

                    plot_path = Path(config["output"]["plot_dir"]) / "MC" / pt
                    plot_path.mkdir(parents=True, exist_ok=True)

                    if config["output"].get("plot_dir"):
                        fig_pass, fig_fail = plot_combined_fit(config, res_mc, plot_path, data_type="MC", sigmoid_eff=sigmoid_eff, args_mass=mass)
                        fig_pass.savefig(plot_path / f"{mc_key}_Pass.png", bbox_inches="tight", dpi=300)
                        fig_fail.savefig(plot_path / f"{mc_key}_Fail.png", bbox_inches="tight", dpi=300)
                        plt.close(fig_pass)
                        plt.close(fig_fail)
                        plt.close("all")

                    output_tables_mc[pt].append(res_mc.get("sum_table", ""))
                    output_progs_mc[pt].append(res_mc.get("sum_prog", ""))
                    output_texts_mc[pt].append(res_mc.get("sum_text", ""))

                    all_results.append(res_mc)

                    eps, err = res_mc["popt"]["epsilon"], res_mc["perr"]["epsilon"]
                    mc_eff, mc_err = eps, err
                    if mc_key is not None:
                        mc_msg = f"{mc_key} fit {'passed' if res_mc['message'].is_valid else 'failed'}"
                    else:
                        mc_msg = "MC N/A"

                    if mc_msg:
                        mc_msg_parts.append(mc_msg)
                    else:
                        mc_msg_parts.append("N/A")

                    style_mc = "green" if res_mc["message"].is_valid else "red"
                    sub_progress.update(task_mc, advance=1, style=style_mc)
                    bins_progress.update(task_bins, advance=1)  # <-- update main bin progress

                elif has_mc:
                    # No MC file for this index, still advance progress bar
                    sub_progress.update(task_mc, advance=1)
                    bins_progress.update(task_bins, advance=1)  # <-- update main bin progress

                data_eff_list.append(data_eff)
                data_err_list.append(data_err)
                mc_eff_list.append(mc_eff)
                mc_err_list.append(mc_err)

                if None not in (data_eff, data_err, mc_eff, mc_err) and mc_eff != 0 and data_eff != 0:
                    sf_val = data_eff / mc_eff
                    rel_err = np.sqrt((data_err / data_eff) ** 2 + (mc_err / mc_eff) ** 2)
                    sf_err_val = sf_val * rel_err
                else:
                    sf_val = None
                    sf_err_val = None

                sf_list.append(sf_val)
                sf_err_list.append(sf_err_val)

            if task_data is not None and task_data in sub_progress.task_ids:
                sub_progress.update(task_data, description=f"    [bold]DATA ({pt}): [{style_data}]{data_key}")
            if task_mc is not None and task_mc in sub_progress.task_ids:
                sub_progress.update(task_mc, description=f"    [bold]MC   ({pt}): [{style_mc}]{mc_key}")

            # After loop
            data_eff_per_bin.append(data_eff_list)
            data_err_per_bin.append(data_err_list)
            mc_eff_per_bin.append(mc_eff_list)
            mc_err_per_bin.append(mc_err_list)
            sf_per_bin.append(sf_list)
            sf_err_per_bin.append(sf_err_list)
            data_msg_per_bin.append(data_msg_parts)
            mc_msg_per_bin.append(mc_msg_parts)
            all_pt_bins.append(pt)

    # Summary logging
    logger.info("\nFit Summary:")
    for pt in all_pt_bins:
        for text_data, text_mc in zip(output_texts_data[pt], output_texts_mc[pt], strict=True):
            console.print(text_data)
            console.print(text_mc)

    # Final SF output summary
    print_efficiency_summary(
        all_pt_bins, data_msg_per_bin, mc_msg_per_bin, data_eff_per_bin, data_err_per_bin, mc_eff_per_bin, mc_err_per_bin, sf_per_bin, sf_err_per_bin
    )

    # If explicit scale_factors are in config, use the per-bin last-seen values (falls back to None)
    if config.get("scale_factors", {}).get("data_mc_pair"):
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Pair Name", style="cyan", justify="right", no_wrap=True)
        table.add_column("Bin", style="green", justify="center", no_wrap=True)
        table.add_column("Efficiency Summary", style="white", justify="left")

        for pair_name, (data_key, mc_key) in config["scale_factors"]["data_mc_pair"].items():
            for pt in all_pt_bins:
                data_eff = data_eff_dict.get(pt, {}).get(data_key)
                data_err = data_err_dict.get(pt, {}).get(data_key)
                mc_eff = mc_eff_dict.get(pt, {}).get(mc_key)
                mc_err = mc_err_dict.get(pt, {}).get(mc_key)

                if None in (data_eff, data_err, mc_eff, mc_err):
                    eff_line = "DATA: N/A | MC: N/A | SF: N/A"
                    table.add_row(pair_name, str(pt), eff_line)
                    continue

                if mc_eff != 0 and data_eff != 0:
                    sf = data_eff / mc_eff
                    rel_err = np.sqrt((data_err / data_eff) ** 2 + (mc_err / mc_eff) ** 2)
                    sf_err = sf * rel_err
                else:
                    sf = None
                    sf_err = None

                if sf is not None:
                    eff_line = f"DATA: {data_eff:.5f} ± {data_err:.5f} | MC: {mc_eff:.5f} ± {mc_err:.5f} | SF: {sf:.5f} ± {sf_err:.5f}"
                else:
                    eff_line = f"DATA: {data_eff:.5f} ± {data_err:.5f} | MC: {mc_eff:.5f} ± {mc_err:.5f} | SF: N/A (MC or DATA=0)"

                table.add_row(pair_name, str(pt), eff_line)

        console.print(
            Panel.fit(
                table,
                title="[bold yellow]Explicit Scale Factors (Per Bin)[/bold yellow]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    main()
