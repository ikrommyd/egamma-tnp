from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

from egamma_tnp.utils.logger_utils import setup_logger

# Define xrootd prefixes for different regions
xrootd_pfx = {
    "Americas": "root://cmsxrootd.fnal.gov/",
    "Eurasia": "root://xrootd-cms.infn.it/",
    "Yolo": "root://cms-xrd-global.cern.ch/",
}


def get_fetcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a JSON file mapping dataset names to file paths.")

    parser.add_argument(
        "-i",
        "--input",
        help="Input dataset definition file to process.",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--where",
        help="Specify the region for xrootd prefix (only for grid mode).",
        default="Eurasia",
        choices=["Americas", "Eurasia", "Yolo"],
    )
    parser.add_argument(
        "-x",
        "--xrootd",
        help="Override xrootd prefix with the one given.",
        default=None,
    )
    parser.add_argument(
        "--dbs-instance",
        dest="instance",
        help="The DBS instance to use for querying datasets (only for grid mode).",
        type=str,
        default="prod/global",
        choices=["prod/global", "prod/phys01", "prod/phys02", "prod/phys03"],
    )
    parser.add_argument(
        "--mode",
        help="Mode of operation: 'grid' to fetch remote datasets or 'local' to fetch local file paths.",
        choices=["grid", "local"],
        default="grid",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include files in subdirectories (only for local mode).",
    )
    parser.add_argument(
        "--file-extension",
        nargs="*",
        help="Filter files by extensions (e.g., .root .txt) (only for local mode). If not specified, all files are included.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit to the first N files of each dataset in sample JSON. If not specified, all files are included.",
    )
    return parser.parse_args()


def read_input_file(input_txt: str, mode: str, logger) -> list[tuple]:
    """
    Read the input text file and parse dataset names and paths.

    :param input_txt: Path to the input text file
    :param mode: Mode of operation ('grid' or 'local') for validation
    :param logger: Logger instance
    :return: List of tuples (dataset-name, dataset-path)
    """
    fset = []
    with open(input_txt) as fp:
        for i, line in enumerate(fp, start=1):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue  # Skip empty lines and comments
            parts = stripped_line.split(None, 1)  # Split by whitespace into at most 2 parts
            if len(parts) != 2:
                logger.warning(f"Line {i} in '{input_txt}' is malformed: '{line.strip()}'")
                continue
            name, path = parts
            if mode == "local":
                # Optionally, you can add more validation for local paths here
                pass
            elif mode == "grid":
                # Optionally, validate grid dataset paths
                pass
            fset.append((name, path, {}))
    return fset


def read_input_yaml(input_yaml: str, mode: str, logger) -> list[tuple]:
    """
    Read the input text file and parse dataset names and paths.

    :param input_txt: Path to the input text file
    :param mode: Mode of operation ('grid' or 'local') for validation
    :param logger: Logger instance
    :return: List of tuples (dataset-name, dataset-path)
    """
    fset = []
    with open(input_yaml) as fp:
        config = yaml.safe_load(fp)
        try:
            datasets = config["datasets"]
        except KeyError:
            logger.error(f"Input YAML file '{input_yaml}' does not contain 'datasets' key.")
            sys.exit(1)
        for dataset in datasets:
            name = dataset.get("name", "")
            if not name:
                logger.error(f"Dataset entry in '{input_yaml}' is missing 'name': {dataset}")
            paths = dataset.get("dataset", [])
            if not paths:
                logger.error(f"Dataset entry in '{input_yaml}' is missing 'dataset' paths: {dataset}")
            metadata = dataset.get("metadata", {})
            fset.append((name, paths, metadata))
    return fset


def get_dataset_dict_grid(fset: Iterable[Iterable[str]], xrd: str, dbs_instance: str, logger, limit) -> dict[str, dict]:
    """
    Fetch file lists for grid datasets using dasgoclient.

    :param fset: Iterable of tuples (dataset-short-name, dataset-path)
    :param xrd: xrootd prefix
    :param dbs_instance: DBS instance for dasgoclient
    :param logger: Logger instance
    :return: Dictionary with the required nested structure
    """
    fdict = {}

    for name, datasets, meta in fset:
        if isinstance(datasets, str):
            datasets_list = [datasets]
        else:
            datasets_list = datasets
        for dataset in datasets_list:
            logger.info(f"Fetching files for dataset '{name}': '{dataset}'")
            private_appendix = "" if not dataset.endswith("/USER") else " instance=prod/phys03"
            try:
                cmd = f"/cvmfs/cms.cern.ch/common/dasgoclient -query='instance={dbs_instance} file dataset={dataset}{private_appendix}'"
                logger.debug(f"Executing command: {cmd}")
                flist = subprocess.check_output(cmd, shell=True, text=True).splitlines()
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed with /cvmfs/cms.cern.ch/common/dasgoclient for dataset '{dataset}': {e}")
                logger.info("Trying with dasgoclient in PATH.")
                try:
                    cmd = f"dasgoclient -query='instance={dbs_instance} file dataset={dataset}{private_appendix}'"
                    logger.debug(f"Executing command: {cmd}")
                    flist = subprocess.check_output(cmd, shell=True, text=True).splitlines()
                except subprocess.CalledProcessError as e:
                    logger.error(f"dasgoclient command failed for dataset '{dataset}': {e}")
                    raise e

            except Exception as e:
                logger.error(f"Unexpected error while fetching files for dataset '{dataset}': {e}")
                raise e

            flist = [xrd + f for f in flist if f.strip()][:limit]

            if name not in fdict:
                fdict[name] = {"files": dict.fromkeys(flist, "Events")}
            else:
                fdict[name]["files"].update(dict.fromkeys(flist, "Events"))
            logger.info(f"Found {len(flist)} files for dataset '{name}'.")
        fdict[name]["metadata"] = meta

    return fdict


def get_dataset_dict_local(fset: Iterable[Iterable[str]], recursive: bool, extensions: list[str], logger, limit) -> dict[str, dict]:
    """
    Collect file lists for local directories.

    :param fset: Iterable of tuples (dataset-short-name, directory-path)
    :param recursive: Whether to search directories recursively
    :param extensions: List of file extensions to filter (case-insensitive)
    :param logger: Logger instance
    :return: Dictionary with the required nested structure
    """
    fdict = {}

    for name, dir_paths, meta in fset:
        if isinstance(dir_paths, str):
            paths = [dir_paths]
        else:
            paths = dir_paths

        for dir_path in paths:
            logger.info(f"Collecting files for local dataset '{name}': '{dir_path}'")
            directory = Path(dir_path)
            if not directory.is_dir():
                logger.error(f"Directory '{dir_path}' does not exist or is not a directory.")
                continue

            pattern = "**/*" if recursive else "*"
            try:
                files = [
                    str(file.resolve())
                    for file in directory.glob(pattern)
                    if file.is_file() and (not extensions or file.suffix.lower() in [ext.lower() for ext in extensions])
                ][:limit]
                if name not in fdict:
                    fdict[name] = {"files": dict.fromkeys(files, "Events")}
                else:
                    fdict[name]["files"].update(dict.fromkeys(files, "Events"))
                logger.info(f"Found {len(files)} files for local dataset '{name}'.")

            except Exception as e:
                logger.error(f"Error while collecting files from directory '{dir_path}': {e}")

        fdict[name]["metadata"] = meta

    return fdict


def main():
    args = get_fetcher_args()

    logger = setup_logger(level="INFO")

    if args.input.endswith(".txt"):
        # Read and parse the input file
        fset = read_input_file(args.input, args.mode, logger)
    elif args.input.endswith(".yaml") or args.input.endswith(".yml"):
        # Read and parse the input YAML file
        fset = read_input_yaml(args.input, args.mode, logger)
    else:
        logger.error("Input file must have a '.txt' or '.yaml' extension and be a text file!")
        sys.exit(1)

    if not fset:
        logger.error(f"No valid entries found in '{args.input}'. Exiting.")
        sys.exit(1)

    logger.info(f"Using the following dataset names and paths: {fset}")

    if args.mode == "grid":
        # Determine xrootd prefix
        xrd = xrootd_pfx.get(args.where, "")
        if args.xrootd:
            xrd = args.xrootd
        logger.info(f"Using xrootd prefix: '{xrd}'")

        # Fetch grid file paths
        fdict = get_dataset_dict_grid(fset, xrd, args.instance, logger, args.limit)

    elif args.mode == "local":
        # Fetch local file paths
        fdict = get_dataset_dict_local(fset, args.recursive, args.file_extension, logger, args.limit)

    # Check if any data was collected
    if not fdict:
        logger.error("No files were collected. Exiting without creating JSON.")
        sys.exit(1)

    # Define output JSON file path
    output_json = Path(args.input).with_suffix(".json")

    # Write the JSON data to the output file
    try:
        with open(output_json, "w") as fp:
            json.dump(fdict, fp, indent=4)
        logger.info(f"Successfully wrote data to JSON file '{output_json}'.")
    except Exception as e:
        logger.error(f"Error writing to JSON file '{output_json}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
