from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import fsspec


def extract_partition_number(filename):
    """Extracts partition number from filenames ending in partX.parquet."""
    filename_only = os.path.basename(filename)  # Extract only the filename
    match = re.search(r"part(\d+)\.parquet$", filename_only)
    return int(match.group(1)) if match else None


def find_parquet_folder(fs, dataset_output_path: str) -> str | None:
    """Finds the first folder inside the dataset directory that contains .parquet files."""
    try:
        subdirs = fs.ls(dataset_output_path, detail=False)  # Get all subdirectories
        for subdir in sorted(subdirs):
            parquet_files = fs.glob(f"{subdir}/*.parquet")
            if parquet_files:
                return subdir  # Return the first subdir containing parquet files
    except FileNotFoundError:
        pass
    return None


def find_missing_partitions(input_json, output_location):
    """Compares expected partitions to existing parquet files and finds missing partitions."""
    fs, token, paths = fsspec.get_fs_token_paths(output_location)

    missing_partitions = {}

    for dataset, data in input_json.items():
        output_folder_name = dataset.lstrip("/").replace("/", "_")
        dataset_output_path = f"{output_location}/{output_folder_name}"

        if not fs.exists(dataset_output_path):
            # If the entire dataset output folder is missing, all steps are missing
            missing_partitions[dataset] = data
            continue

        parquet_folder = find_parquet_folder(fs, dataset_output_path)
        if parquet_folder is None:
            # If no parquet folder exists, all steps are missing
            missing_partitions[dataset] = data
            continue

        # Get existing partitions from output
        existing_partitions = {
            extract_partition_number(Path(file).name)
            for file in fs.glob(f"{parquet_folder}/*.parquet")
            if extract_partition_number(Path(file).name) is not None
        }

        # Determine total number of expected partitions
        all_steps = []
        for file_info in data.get("files", {}).values():
            all_steps.extend(file_info["steps"])

        max_partition_digits = len(str(len(all_steps) - 1))

        # Find missing partitions
        dataset_missing_partitions = {}

        partition_idx = 0  # Track partition numbers sequentially across files
        for file_path, file_info in data.get("files", {}).items():
            file_missing_partitions = []
            for step in file_info["steps"]:
                expected_partition = f"{partition_idx:0{max_partition_digits}d}"
                if int(expected_partition) not in existing_partitions:
                    file_missing_partitions.append(step)
                partition_idx += 1

            if file_missing_partitions:
                # Preserve all existing fields, only update 'steps'
                dataset_missing_partitions[file_path] = {
                    **file_info,  # Keep all existing fields
                    "steps": file_missing_partitions,  # Only update missing steps
                }

        if dataset_missing_partitions:
            missing_partitions[dataset] = {**data, "files": dataset_missing_partitions}

    return missing_partitions


def main():
    parser = argparse.ArgumentParser(description="Find missing parquet partitions in datasets.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output-location", type=str, required=True, help="Base output location for checking partitions.")
    parser.add_argument("--output-json", type=str, default="missing_partitions.json", help="Path to save missing partitions JSON.")
    args = parser.parse_args()

    # Load input JSON
    with fsspec.open(args.input_json, "r") as f:
        input_json = json.load(f)

    # Find missing partitions
    missing_partitions = find_missing_partitions(input_json, args.output_location)

    # Save output
    with fsspec.open(args.output_json, "w") as f:
        json.dump(missing_partitions, f, indent=2)


if __name__ == "__main__":
    main()
