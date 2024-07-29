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
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as file:
            return json.load(file)
    else:
        with open(file_path) as file:
            return json.load(file)


def set_binning(binning_config):
    for key, bins in binning_config.items():
        egamma_tnp.binning.set(key, bins)


def filter_class_args(class_, args):
    sig = inspect.signature(class_.__init__)
    return {k: v for k, v in args.items() if k in sig.parameters}


def initialize_class(config, args, fileset):
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
    results = {}
    for method in methods:
        method_name = method["name"]
        method_args = method["args"]
        method_to_call = getattr(instance, method_name)
        for arg in method_args:
            if arg in ["compute", "scheduler", "progress"]:
                raise ValueError(f"Argument `{arg}` is not allowed to be specified in the JSON configuration file.")
        if method_name != "get_tnp_arrays" and isinstance(method_args["filter"], list):
            new_method_args = method_args.copy()
            del new_method_args["filter"]
            results[method_name] = {f: method_to_call(compute=False, filter=f, **new_method_args) for f in method_args["filter"]}
        else:
            results[method_name] = method_to_call(compute=False, **method_args)
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
        # stat is 0 the proxy is valid
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
    --tags_pt_cut: int or float, optional
        The Pt cut to apply to the tag particles. The default is 35.
    --probes_pt_cut: int or float, optional
        The Pt threshold of the probe particles to apply in the calculations. The default is None.
    --tags_abseta_cut: int or float, optional
        The absolute Eta cut to apply to the tag particles. The default is 2.5.
    --probes_abseta_cut: int or float, optional
        The absolute Eta cut to apply to the probe particles. The default is 2.5.
    --cutbased_id: str, optional
        ID expression to apply to the probes. An example is "cutBased >= 2". If None, no cutbased ID is applied. The default is None.
    --extra_tags_mask: str, optional
        An extra mask to apply to the tags. The default is None. Must be of the form "events.<mask> & events.<mask> & ...".
    --extra_probes_mask: str, optional
        An extra mask to apply to the probes. The default is None. Must be of the form "events.<mask> & events.<mask> & ...".
    --goldenjson: str, optional
        The golden JSON to use for luminosity masking. The default is None.
    --extra_filter: Callable, optional
        An extra function to filter the events. The default is None. Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
    --extra_filter_args: dict, optional
        Extra arguments to pass to the extra filter. The default is {}.
    --use_sc_eta: bool, optional
        Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
    --use_sc_phi: bool, optional
        Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
    --avoid_ecal_transition_tags: bool, optional
        Avoid the ECAL transition region for the tags with an Eta cut. The default is True.
    --avoid_ecal_transition_probes: bool, optional
        Avoid the ECAL transition region for the probes with an Eta cut. The default is False.
    --require_event_to_pass_hlt_filter: bool, optional
        Require the event to pass the HLT filter under study to consider a probe belonging to that event as passing. The default is True.
    --start_from_diphotons: bool, optional
        Consider photon-photon pairs as tag-probe pairs. The default is True.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser with all the specified arguments.
    """
    parser = argparse.ArgumentParser(description="Common argument parser for Tag and Probe calculations")

    # Configuration
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON configuration file specifying the class and methods to run. Default is None.")

    # Fileset
    parser.add_argument("--fileset", type=str, required=True, help="The fileset to perform the tag and probe calculations on.")

    # Binning
    parser.add_argument("--binning", type=str, help="Path to a JSON file specifying the binning. Default is None.")

    # Executor
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
    parser.add_argument("--jobflavour", type=str, default="longlunch", help="Job flavour for lxplus conodr job submission")
    parser.add_argument("--queue", type=str, help="Queue for job submission")
    parser.add_argument("--walltime", type=str, help="Walltime for job execution")

    # Common options
    parser.add_argument("--tags_pt_cut", type=float, default=35, help="The Pt cut to apply to the tag particles. Default is 35.")
    parser.add_argument("--probes_pt_cut", type=float, help="The Pt threshold of the probe particles to apply in the calculations. Default is None.")
    parser.add_argument("--tags_abseta_cut", type=float, default=2.5, help="The absolute Eta cut to apply to the tag particles. Default is 2.5.")
    parser.add_argument("--probes_abseta_cut", type=float, default=2.5, help="The absolute Eta cut to apply to the probe particles. Default is 2.5.")
    parser.add_argument("--cutbased_id", type=str, help='ID expression to apply to the probes. Example: "cutBased >= 2". Default is None.')
    parser.add_argument("--extra_tags_mask", type=str, help="An extra mask to apply to the tags. Default is None.")
    parser.add_argument("--extra_probes_mask", type=str, help="An extra mask to apply to the probes. Default is None.")
    parser.add_argument("--goldenjson", type=str, help="The golden JSON to use for luminosity masking. Default is None.")
    parser.add_argument("--extra_filter", type=str, help="An extra function to filter the events. Default is None.")
    parser.add_argument("--extra_filter_args", type=dict, default={}, help="Extra arguments to pass to the extra filter. Default is {}.")
    parser.add_argument("--use_sc_eta", type=bool, default=False, help="Use the supercluster Eta instead of the Eta from the primary vertex. Default is False.")
    parser.add_argument("--use_sc_phi", type=bool, default=False, help="Use the supercluster Phi instead of the Phi from the primary vertex. Default is False.")
    parser.add_argument(
        "--avoid_ecal_transition_tags", type=bool, default=True, help="Avoid the ECAL transition region for the tags with an Eta cut. Default is True."
    )
    parser.add_argument(
        "--avoid_ecal_transition_probes", type=bool, default=False, help="Avoid the ECAL transition region for the probes with an Eta cut. Default is False."
    )
    parser.add_argument(
        "--require_event_to_pass_hlt_filter",
        type=bool,
        default=True,
        help="Require the event to pass the HLT filter under study to consider a probe belonging to that event as passing. Default is True.",
    )
    parser.add_argument("--start_from_diphotons", type=bool, default=True, help="Consider photon-photon pairs as tag-probe pairs. Default is True.")

    return parser
