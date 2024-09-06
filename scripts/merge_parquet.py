from __future__ import annotations

import argparse
import logging
import os

import awkward as ak

logger = logging.getLogger(__name__)


def generate(files):
    for f in files:
        array = ak.from_parquet(f)
        yield array
        del array


def main():
    parser = argparse.ArgumentParser(description="Simple utility script to merge all parquet files in one folder.")
    parser.add_argument("--source", type=str, required=True, help="Source folder containing parquet files.")
    parser.add_argument("--target", type=str, required=True, help="Target parquet file location.")
    args = parser.parse_args()

    logger.info(f"Merging parquet files from {args.source} to {args.target}")
    files = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.endswith(".parquet")]

    ak.to_parquet_row_groups(generate(files), args.target)
    logger.info("Done.")


if __name__ == "__main__":
    main()
