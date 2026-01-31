from __future__ import annotations

import logging

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()

LOGGER_NAME = "egamma_tnp"


def setup_logger(level: str = "INFO", logfile: str | None = None, time: bool | None = False) -> logging.Logger:
    """Setup a logger that uses RichHandler to write the same message both in stdout
    and in a log file called logfile. Level of information can be customized and
    dumping a logfile is optional.

    :param level: level of information
    :type level: str, optional
    :param logfile: file where information are stored
    :type logfile: str
    """
    logger = logging.getLogger(LOGGER_NAME)  # need to give it a name, otherwise *way* too much info gets printed out from e.g. numba
    logging.getLogger().handlers.clear()

    # Set up level of information
    possible_levels = ["INFO", "DEBUG"]
    if level not in possible_levels:
        raise ValueError("Passed wrong level for the logger. Allowed levels are: {}".format(", ".join(possible_levels)))
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter("%(message)s")
    if time:
        formatter = logging.Formatter("%(asctime)s %(message)s")

    # Set up stream handler (for stdout)
    stream_handler = RichHandler(show_time=False, rich_tracebacks=True)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Set up file handler (for logfile)
    if logfile:
        file_handler = RichHandler(
            show_time=False,
            rich_tracebacks=True,
            console=Console(file=open(logfile, "w")),
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


COLOR_BORDER = "#00B4D8"
COLOR_PRIMARY = "#E6EDF3"
COLOR_SECONDARY = "#AAB3BF"
COLOR_SUCCESS = "#06D6A0"
COLOR_ERROR = "#E63946"
COLOR_HIGHLIGHT = "#FFB703"
COLOR_WARNING = "#F77F00"
COLOR_BG_DARK = "#0D1117"


class CustomTimeElapsedColumn(TimeElapsedColumn):
    """Show time elapsed while running; show completion time when done."""

    def render(self, task):
        if task.finished:
            # If we haven't stored the completion time yet, store it
            if not hasattr(task, "completion_time"):
                task.completion_time = task.elapsed
            if task.completed == 0 and task.total > 0:
                return Text("No files to fit", style="yellow")
            return Text(f"Completed! ({task.completion_time:.1f}s)", style="green")
        else:
            return Text(f"{task.elapsed:.1f}s", style="progress.elapsed")


def print_efficiency_summary(
    all_pt_bins, data_msg_per_bin, mc_msg_per_bin, data_eff_per_bin, data_err_per_bin, mc_eff_per_bin, mc_err_per_bin, sf_per_bin, sf_err_per_bin
):
    for pt, data_msg, mc_msg, data_list, data_err_list, mc_list, mc_err_list, sf_list, sf_err_list in zip(
        all_pt_bins,
        data_msg_per_bin,
        mc_msg_per_bin,
        data_eff_per_bin,
        data_err_per_bin,
        mc_eff_per_bin,
        mc_err_per_bin,
        sf_per_bin,
        sf_err_per_bin,
        strict=True,
    ):
        table = Table(show_header=True, header_style=f"bold {COLOR_HIGHLIGHT}", box=box.ROUNDED)
        table.add_column("DATA status", style=COLOR_BORDER, justify="right", no_wrap=True)
        table.add_column("MC status", style=COLOR_BORDER, justify="right", no_wrap=True)
        table.add_column("Data Efficiency", style=COLOR_PRIMARY, justify="left")
        table.add_column("MC Efficiency", style=COLOR_PRIMARY, justify="left")
        table.add_column("SF", style=COLOR_PRIMARY, justify="left")

        max_len = max(len(data_list), len(mc_list))
        for i in range(max_len):
            data_str = f"DATA: {data_list[i]:.4f} ± {data_err_list[i]:.4f}" if data_list[i] is not None else "DATA: N/A"
            mc_str = f"MC: {mc_list[i]:.4f} ± {mc_err_list[i]:.4f}" if mc_list[i] is not None else "MC: N/A"
            data_text = data_msg[i] if i < len(data_msg) else "N/A"
            mc_text = mc_msg[i] if i < len(mc_msg) else "N/A"

            sf_str = f"SF: {sf_list[i]:.5f} ± {sf_err_list[i]:.5f}" if sf_list[i] is not None else "SF: N/A (missing DATA or MC)"

            status_data = Text()
            status_data.append("DATA: ", style=f"bold {COLOR_PRIMARY}")
            status_data.append(f"{data_text}", style=COLOR_SUCCESS if "passed" in data_text else COLOR_ERROR)

            status_mc = Text()
            status_mc.append("MC: ", style=f"bold {COLOR_PRIMARY}")
            status_mc.append(f"{mc_text}", style=COLOR_SUCCESS if "passed" in mc_text else COLOR_ERROR)

            table.add_row(status_data, status_mc, data_str, mc_str, sf_str)

        console.print(
            Panel.fit(
                table,
                title=f"[bold {COLOR_HIGHLIGHT}]pT Bin: {pt}[/bold {COLOR_HIGHLIGHT}]",
                border_style=COLOR_BORDER,
                padding=(1, 2),
            )
        )


def print_fit_summary_rich(
    m, popt, perr, edges_pass, args_bin, BINS_INFO, Pearson_chi2, Poisson_chi2, total_ndof, args_data, fit_type, sigmoid_eff=False, DATA_NAME=None, MC_NAME=None
):
    tol = 1e-8
    params_at_limit = []
    for p in m.params:
        if p.lower_limit is not None and abs(p.value - p.lower_limit) < tol * max(1.0, abs(p.lower_limit)):
            params_at_limit.append(p.name)
        if p.upper_limit is not None and abs(p.value - p.upper_limit) < tol * max(1.0, abs(p.upper_limit)):
            params_at_limit.append(p.name)
    any_at_limit = bool(params_at_limit)

    fmin = m.fmin
    valid_min = bool(m.valid)

    above_edm = getattr(fmin, "is_above_max_edm", False)
    reached_call_limit = getattr(fmin, "has_reached_call_limit", False)
    hesse_failed = getattr(fmin, "hesse_failed", False)
    cov_accurate = getattr(fmin, "has_accurate_covar", False)

    # --- Bin Info Panel ---
    bin_box = Table.grid(padding=(0, 1), expand=True)
    bin_box.add_column(justify="right", style=COLOR_PRIMARY)
    bin_box.add_column(justify="left", style=COLOR_PRIMARY)
    bin_box.add_row("Bin:", str(args_bin))
    if args_data is not None:
        bin_box.add_row("DATA:", f"[{COLOR_PRIMARY}]{DATA_NAME}")
    if MC_NAME is not None:
        bin_box.add_row("MC:", f"[{COLOR_PRIMARY}]{MC_NAME}")

    panel_bin_info = Panel(bin_box, title=f"[b {COLOR_PRIMARY}]Bin Info", box=box.ROUNDED, border_style=COLOR_BORDER)

    # --- Status Panel ---
    def status_item(ok: bool, label: str):
        symbol = f"[{COLOR_SUCCESS}]✅" if ok else f"[{COLOR_ERROR}]❌"
        return f"{symbol} {label}"

    status_table = Table.grid(padding=(0, 1), expand=True)
    status_table.add_column(style=COLOR_PRIMARY)
    status_table.add_column(justify="right", style=COLOR_PRIMARY)
    status_table.add_column(justify="left", style=COLOR_PRIMARY)
    status_table.add_row(status_item(valid_min, "Valid Minimum"))
    status_table.add_row(status_item(not any_at_limit, "No Parameters at Limit"))
    status_table.add_row(status_item(not above_edm, "Below EDM Threshold"))
    status_table.add_row(status_item(not reached_call_limit, "Below Call Limit"))
    status_table.add_row(status_item(not hesse_failed, "Hesse OK"))
    status_table.add_row(status_item(cov_accurate, "Covariance OK"))

    panel_fit_checks = Panel(status_table, title=f"[b {COLOR_PRIMARY}]Fit Checks", box=box.ROUNDED, border_style=COLOR_BORDER)

    # --- Goodness of Fit ---
    gof_table = Table(show_header=False, show_lines=True, padding=(0, 1), box=box.ROUNDED, expand=True)
    gof_table.add_column("", justify="right", ratio=1, style=COLOR_PRIMARY)
    gof_table.add_column("", justify="center", ratio=2, style=COLOR_PRIMARY)

    pearson_chi2_ndf = Pearson_chi2 / total_ndof if total_ndof > 0 else float("nan")
    poisson_chi2_ndf = Poisson_chi2 / total_ndof if total_ndof > 0 else float("nan")

    gof_table.add_row("Pearson χ²/NDF:", f"{pearson_chi2_ndf:.3f}")
    gof_table.add_row("Poisson χ²/NDF:", f"{poisson_chi2_ndf:.3f}")

    panel_gof = Panel(gof_table, title=f"[b {COLOR_PRIMARY}]Goodness of Fit", box=box.ROUNDED, border_style=COLOR_BORDER)

    # --- Fit Params ---
    param_table = Table(show_lines=True, padding=(0, 1), box=box.ROUNDED, expand=True, header_style=f"bold {COLOR_HIGHLIGHT}")
    param_table.add_column("Fit Parameters", justify="right", ratio=1, style=COLOR_PRIMARY)
    param_table.add_column("Value ± Error", justify="center", ratio=3, style=COLOR_PRIMARY)

    def add_param_row(key, val_fmt="{:.4f}", err_fmt="{:.4f}", label=None):
        if key in popt:
            label = label or key
            val = popt[key]
            err = perr.get(key, float("nan"))
            param_table.add_row(label, f"{val_fmt.format(val)} ± {err_fmt.format(err)}")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    add_param_row("N", "{:.1f}", "{:.1f}")
    if "epsilon" in popt:
        if sigmoid_eff:
            eff = sigmoid(popt["epsilon"])
            eff_err = abs(perr.get("epsilon", 0) * eff * (1 - eff))
            param_table.add_row("ε (sigmoid)", f"{eff:.6f} ± {eff_err:.6f}")
        else:
            add_param_row("epsilon", "{:.6f}", "{:.6f}", "ε")
    add_param_row("B_p", "{:.1f}", "{:.1f}")
    add_param_row("B_f", "{:.1f}", "{:.1f}")

    panel_params = Panel(param_table, title=f"[b {COLOR_PRIMARY}]Fit Parameters", box=box.ROUNDED, border_style=COLOR_BORDER)

    # --- Full Parameter Table ---
    full_param_table = Table(show_lines=True, header_style=f"bold {COLOR_HIGHLIGHT}", padding=(0, 1), box=box.ROUNDED, expand=True)
    full_param_table.add_column("Id", justify="right", style=COLOR_SECONDARY)
    full_param_table.add_column("Name", style=COLOR_PRIMARY)
    full_param_table.add_column("Value", justify="right", style=COLOR_PRIMARY)
    full_param_table.add_column("Error", justify="right", style=COLOR_PRIMARY)
    full_param_table.add_column("MINOS -", justify="right", style=COLOR_PRIMARY)
    full_param_table.add_column("MINOS +", justify="right", style=COLOR_PRIMARY)
    full_param_table.add_column("Fixed", justify="center", style=COLOR_PRIMARY)
    full_param_table.add_column("Lower", justify="right", style=COLOR_PRIMARY)
    full_param_table.add_column("Upper", justify="right", style=COLOR_PRIMARY)

    for i, p in enumerate(m.params):
        low = str(p.lower_limit) if p.lower_limit is not None else ""
        high = str(p.upper_limit) if p.upper_limit is not None else ""
        if p.is_fixed:
            err_str, minos_lower, minos_upper = "fixed", "", ""
        else:
            err_str = f"{p.error:.3f}"
            merr = m.merrors.get(p.name, None)
            minos_lower = f"{merr.lower:.3f}" if merr else ""
            minos_upper = f"{merr.upper:.3f}" if merr else ""

        full_param_table.add_row(str(i), p.name, f"{p.value:.3f}", err_str, minos_lower, minos_upper, str(p.is_fixed), low, high)

    panel_table = Panel(full_param_table, title=f"[b {COLOR_PRIMARY}] Full Parameter Details", box=box.ROUNDED, border_style=COLOR_BORDER)

    # --- Layout ---
    left_column = Group(panel_bin_info, panel_fit_checks, panel_gof, panel_params)
    right_column = panel_table
    columns = Columns([left_column, right_column], equal=True, expand=True)
    return columns
