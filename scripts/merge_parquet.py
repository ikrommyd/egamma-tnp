from __future__ import annotations

import argparse

import awkward as ak
import fsspec

from egamma_tnp.utils.logger_utils import setup_logger

logger = setup_logger(level="INFO")


def generate(files):
    for f in files:
        array = ak.from_parquet(f)
        yield array
        del array


def main():
    parser = argparse.ArgumentParser(description="Simple utility script to merge all parquet files in one folder.")
    parser.add_argument("--source", type=str, required=True, help="Source folder or path of files with wildcards containing parquet files.")
    parser.add_argument("--target", type=str, required=True, help="Target parquet file location.")
    args = parser.parse_args()

    logger.info(f"Merging parquet files from {args.source} to {args.target}")
    fs, token, paths = fsspec.get_fs_token_paths(args.source)
    if len(paths) == 1:
        if fs.isfile(paths[0]):
            files = paths
        else:
            files = fs.glob(f"{paths[0]}/*.parquet")
    else:
        files = paths

    final_files = [fs.unstrip_protocol(f) for f in files]
    ak.to_parquet_row_groups(generate(final_files), args.target, extensionarray=True)
    logger.info("Done.")


if __name__ == "__main__":
    main()
