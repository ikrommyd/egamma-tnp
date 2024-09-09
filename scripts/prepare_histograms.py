from __future__ import annotations

import argparse
import pickle

import fsspec

from egamma_tnp.utils import create_hists_root_file_for_fitter
from egamma_tnp.utils.logger_utils import setup_logger

logger = setup_logger(level="INFO")


def main():
    parser = argparse.ArgumentParser(description="Simple utility script to prepare the histograms for the fitter.")
    parser.add_argument("--source", type=str, required=True, help="Source histograms pickle file location.")
    parser.add_argument("--target", type=str, required=True, help="Target histograms ROOT file location.")
    parser.add_argument("--binning", type=str, required=True, help="Target binning pickle file location.")
    parser.add_argument(
        "--axes",
        nargs="+",
        type=str,
        required=False,
        default=["el_eta", "el_pt"],
        help="Axes to be used for the histograms. Only used when converting N-dimensional mll histograms. Default is ['el_eta', 'el_pt']",
    )
    args = parser.parse_args()

    logger.info(f"Preparing histograms for fitter from {args.source} to {args.target.replace('.root', '*.root')}.")
    with fsspec.open(args.source, "rb") as f:
        histos = pickle.load(f)
    create_hists_root_file_for_fitter(histos, args.target, args.binning, args.axes)

    logger.info("Done.")


if __name__ == "__main__":
    main()
