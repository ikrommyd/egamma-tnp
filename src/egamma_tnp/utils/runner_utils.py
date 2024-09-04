from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import pickle
import subprocess
import warnings

import awkward as ak
import dask_awkward as dak
import fsspec

import egamma_tnp
from egamma_tnp import (
    ElectronTagNProbeFromNanoAOD,
    ElectronTagNProbeFromNTuples,
    PhotonTagNProbeFromNanoAOD,
    PhotonTagNProbeFromNTuples,
)


def load_json(file_path):
    """Load a JSON file from the given file path, supporting both plain and gzipped JSON files."""
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as file:
            return json.load(file)
    else:
        with open(file_path) as file:
            return json.load(file)


def load_settings(settings_path):
    """
    Load settings from a specified JSON file.
    If the settings_path is not provided, load from the default settings file.
    """
    if settings_path is None:
        # Default settings JSON path
        settings_path = os.path.join(os.path.dirname(__file__), "default_runner_settings.json")

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    return load_json(settings_path)


def parse_uproot_options(uproot_options):
    """Convert uproot_options from JSON format to the proper Python format."""
    if uproot_options is None:
        return {}

    if isinstance(uproot_options, dict):
        if "allow_read_errors_with_report" in uproot_options:
            errors = uproot_options["allow_read_errors_with_report"]
            if errors is True:  # If set to true, keep it as True
                uproot_options["allow_read_errors_with_report"] = True
            elif isinstance(errors, list):
                uproot_options["allow_read_errors_with_report"] = tuple(__builtins__[error] for error in errors)

    return uproot_options


def merge_settings_with_args(args, settings):
    """
    Merge settings from the JSON file with the command-line arguments.
    Settings from JSON will be added only if they don't already exist in args.
    """
    # Add all settings from the JSON to the args, only if not already present
    for key, value in settings.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    return args


def set_binning(binning_config):
    """Set binning configuration using the provided JSON data."""
    for key, bins in binning_config.items():
        egamma_tnp.binning.set(key, bins)


def filter_class_args(class_, args):
    """Filter out arguments that are not in the class's __init__ signature."""
    sig = inspect.signature(class_.__init__)
    return {k: v for k, v in args.items() if k in sig.parameters}


def initialize_class(config, args, fileset):
    """Initialize the appropriate Tag and Probe class based on the workflow specified in the config."""
    class_map = {
        "ElectronTagNProbeFromNanoAOD": ElectronTagNProbeFromNanoAOD,
        "ElectronTagNProbeFromNTuples": ElectronTagNProbeFromNTuples,
        "PhotonTagNProbeFromNanoAOD": PhotonTagNProbeFromNanoAOD,
        "PhotonTagNProbeFromNTuples": PhotonTagNProbeFromNTuples,
    }
    class_name = config["workflow"]
    workflow = class_map[class_name]
    class_args = config["workflow_args"] | filter_class_args(workflow, vars(args))
    class_args.pop("fileset")
    return workflow(fileset=fileset, **class_args)


def run_methods(instance, methods):
    """Run specified methods on the initialized Tag and Probe instance."""
    results = []
    for method in methods:
        method_name = method["name"]
        method_args = method["args"]
        method_args["uproot_options"] = parse_uproot_options(method_args["uproot_options"])
        method_to_call = getattr(instance, method_name)

        # Check for disallowed arguments in the JSON configuration
        for arg in method_args:
            if arg in ["compute", "scheduler", "progress"]:
                raise ValueError(f"Argument `{arg}` is not allowed to be specified in the JSON configuration file.")

        # Handle methods with a list of filters
        if method_name != "get_tnp_arrays":
            new_method_args = method_args.copy()
            del new_method_args["filter"]
            modified_filters = [method_args["filter"]] if isinstance(method_args["filter"], str) else method_args["filter"]
            result = {f: method_to_call(compute=False, filter=f, **new_method_args) for f in modified_filters}
        else:
            result = method_to_call(compute=False, **method_args)

        # Append the result, method name, and args to the results list
        results.append({"method": method_name, "args": method_args, "result": result})

    return results


def save_array_to_parquet(array, output_dir, dataset, subdir, prefix=None, repartition_n=5):
    """Helper function to save a Dask array to a Parquet file."""
    # Ensure output directory is set
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, dataset.removeprefix("/").replace("/", "_"), subdir)

    # Repartition the array if needed
    if repartition_n:
        array = array.repartition(n_to_one=repartition_n)

    return dak.to_parquet(array, output_path, compute=False, prefix=prefix)


def process_to_compute(to_compute, output_dir, repartition_n=5):
    """
    Process the task graph (to_compute) to save arrays to Parquet files and keep track of reports.

    Parameters:
    - to_compute (list): The task graph to process, which includes method results and arguments.
    - output_dir (str): The directory to save the output Parquet files. If None, current directory is used.
    - repartition_n (int): The number of partitions to reduce to during saving. Default is 5.

    Returns:
    - processed_to_compute (list): The modified task graph with arrays replaced by Parquet save tasks.
    """
    processed_to_compute = []
    method_counts = {}

    for entry in to_compute:
        method = entry["method"]
        args = entry["args"]
        result = entry["result"]
        processed_result = {}

        # Track how many times each method is called
        method_counts[method] = method_counts.get(method, 0) + 1
        subdir_name = f"{method}_{method_counts[method]}"

        # Separate arrays and reports if present
        if isinstance(result, tuple):
            arrays, reports = result
        else:
            arrays = result
            reports = None

        # Handle 'get_tnp_arrays' method
        if method == "get_tnp_arrays":
            for dataset, array in arrays.items():
                processed_result[dataset] = save_array_to_parquet(array, output_dir, dataset, subdir_name, prefix="NTuples", repartition_n=repartition_n)

        # Handle 'get_passing_and_failing_probes' method
        elif method == "get_passing_and_failing_probes":
            for filter_name, datasets in arrays.items():
                processed_result[filter_name] = {}
                for dataset, arr_dict in datasets.items():
                    processed_result[filter_name][dataset] = {}
                    for key in arr_dict:  # 'passing' and 'failing'
                        prefix = f"{key}_{filter_name.replace(' ', '_').replace('>=', 'gte').replace('<=', 'lte').replace('>','gt').replace('<','lt')}_histos"
                        processed_result[filter_name][dataset][key] = save_array_to_parquet(
                            arr_dict[key], output_dir, dataset, subdir_name, prefix=prefix, repartition_n=repartition_n
                        )

        # For other methods, just keep the result as is
        else:
            processed_result = arrays

        # Add reports if present
        if reports is not None:
            processed_result["reports"] = reports

        # Append to the list of processed tasks
        processed_to_compute.append({"method": method, "args": args, "result": processed_result})

    return processed_to_compute


def save_histogram_dict_to_pickle(hist_dict, output_dir, dataset, subdir, filename):
    """Helper function to save a dictionary of histograms to a Pickle file."""
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, dataset.removeprefix("/").replace("/", "_"), subdir).removeprefix("simplecache::")

    with fsspec.open(os.path.join(output_path, f"{filename}.pkl"), "wb") as f:
        pickle.dump(hist_dict, f)


def save_report_to_json(report, output_dir, dataset, subdir):
    """Helper function to save a report to a JSON file."""
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, dataset.removeprefix("/").replace("/", "_"), subdir).removeprefix("simplecache::")
    ak.to_json(report, os.path.join(output_path, "report.json"), num_readability_spaces=1, num_indent_spaces=4)


def process_out(out, output_dir):
    """
    Process the output after computing the task graph.
    This function saves histograms to pickle files and reports to JSON files.

    Parameters:
    - out (list): The computed output from Dask, containing method results and arguments.
    - output_dir (str): The directory to save the output files. If None, the current directory is used.
    """
    method_counts = {}  # Initialize method counts

    for entry in out:
        method = entry["method"]
        result = entry["result"]

        # Track how many times each method is called
        method_counts[method] = method_counts.get(method, 0) + 1
        subdir_name = f"{method}_{method_counts[method]}"

        # Handle the report saving
        reports = result.pop("reports", None)
        if reports:
            for dataset, report in reports.items():
                save_report_to_json(report, output_dir, dataset, subdir_name)

        if method in ["get_1d_pt_eta_phi_tnp_histograms", "get_nd_tnp_histograms"]:
            for filter_name, datasets in result.items():
                for dataset, hist_dict in datasets.items():
                    filename = f"{filter_name.replace(' ', '_').replace('>=', 'gte').replace('<=', 'lte').replace('>','gt').replace('<','lt')}_histos"
                    save_histogram_dict_to_pickle(hist_dict, output_dir, dataset, subdir_name, filename)

        else:
            continue

    return out


def get_proxy():
    """
    Use voms-proxy-info to check if a proxy is available.
    If so, copy it to $HOME/.proxy and return the path.
    An exception is raised in the following cases:
    - voms-proxy-info is not installed
    - the proxy is not valid

    :return: Path to proxy
    :rtype: str
    """
    if subprocess.getstatusoutput("voms-proxy-info")[0] != 0:
        warnings.warn("voms-proxy-init not found, You need a valid certificate to access data over xrootd.", stacklevel=1)
    else:
        stat, out = subprocess.getstatusoutput("voms-proxy-info -e -p")
        # stat is 0 if the proxy is valid
        if stat != 0:
            raise RuntimeError("No valid proxy found. Please create one.")

        _x509_localpath = out
        _x509_path = os.environ["HOME"] + f'/.{_x509_localpath.split("/")[-1]}'
        os.system(f"cp {_x509_localpath} {_x509_path}")

        return _x509_path


def get_main_parser():
    """
    Common argument parser for Tag and Probe calculations.

    This parser includes all the necessary arguments for performing generic tag
    and probe calculations for both electrons and photons from NanoAOD and NTuples.

    Parameters
    ----------
    --config: str
        Path to a JSON configuration file specifying the class and methods to run.
    --settings: str, optional
        Path to a JSON file specifying common options. The default is None.
    --fileset: str
        The fileset to perform the tag and probe calculations on.
    --binning: str, optional
        Path to a JSON file specifying the binning. The default is None.
    --output: str, optional
        Path to the output directory. The default is the current working directory.
    --executor: str, optional
        The executor to use for the computations. The default is None and lets dask decide.
    --cores: int, optional
        Number of cores for each worker. The default is None.
    --memory: str, optional
        Memory allocation for each worker. The default is None.
    --disk: str, optional
        Disk allocation for each worker. The default is None.
    --scaleout: int, optional
        Maximum number of workers. The default is None.
    --adaptive: bool, optional
        Adaptive scaling. The default is True.
    --voms: str, optional
        Path to the VOMS proxy. The default is None. If not specified, it will try to find if there is a valid proxy available.
    --port: int, optional
        Port for the Dask scheduler. The default is 8786.
    --dashboard_address: str, optional
        Address for the Dask dashboard. The default is ":8787".
    --jobflavour: str, optional
        Job flavour for job submission. The default is "longlunch".
    --queue: str, optional
        Queue for job submission. The default is None.
    --walltime: str, optional
        Walltime for job execution. The default is None.
    --log_directory: str, optional
        Directory to save dask worker logs. The default is None.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser with all the specified arguments.
    """
    parser = argparse.ArgumentParser(description="Common argument parser for Tag and Probe calculations")

    # Configuration
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON configuration file specifying the class and methods to run. Default is None.")
    parser.add_argument("--settings", type=str, help="Path to a JSON file specifying common options. Default is None.")
    parser.add_argument("--fileset", type=str, required=True, help="The fileset to perform the tag and probe calculations on.")
    parser.add_argument("--binning", type=str, help="Path to a JSON file specifying the binning. Default is None.")
    parser.add_argument("--output", type=str, help="Path to the output directory. Default is None.")
    parser.add_argument("--executor", type=str, help="The executor to use for the computations. Default is None and lets dask decide.")
    parser.add_argument("--cores", type=int, help="Number of cores for each worker")
    parser.add_argument("--memory", type=str, help="Memory allocation for each worker")
    parser.add_argument("--disk", type=str, help="Disk allocation for each worker")
    parser.add_argument("--scaleout", type=int, help="Maximum number of workers")
    parser.add_argument("--adaptive", type=bool, default=True, help="Adaptive scaling")
    parser.add_argument(
        "--voms", type=str, help="Path to the VOMS proxy. Default is None. If not specified, it will try to find if there is a valid proxy available."
    )
    parser.add_argument("--port", type=int, default=8786, help="Port for the Dask scheduler")
    parser.add_argument("--dashboard_address", type=str, default=":8787", help="Address for the Dask dashboard")
    parser.add_argument("--jobflavour", type=str, default="longlunch", help="Job flavour for lxplus condor job submission")
    parser.add_argument("--queue", type=str, help="Queue for job submission")
    parser.add_argument("--walltime", type=str, help="Walltime for job execution")
    parser.add_argument("--log_directory", type=str, help="Directory to save dask worker logs")

    return parser
