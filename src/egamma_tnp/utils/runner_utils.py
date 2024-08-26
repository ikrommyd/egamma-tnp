from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import subprocess
import warnings

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
        method_to_call = getattr(instance, method_name)

        # Check for disallowed arguments in the JSON configuration
        for arg in method_args:
            if arg in ["compute", "scheduler", "progress"]:
                raise ValueError(f"Argument `{arg}` is not allowed to be specified in the JSON configuration file.")

        # Handle methods with a list of filters
        if method_name != "get_tnp_arrays" and isinstance(method_args.get("filter"), list):
            new_method_args = method_args.copy()
            del new_method_args["filter"]
            result = {f: method_to_call(compute=False, filter=f, **new_method_args) for f in method_args["filter"]}
        else:
            result = method_to_call(compute=False, **method_args)

        # Append the result, method name, and args to the results list
        results.append({"method": method_name, "args": method_args, "result": result})

    return results


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

    return parser
