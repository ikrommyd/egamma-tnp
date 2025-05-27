from __future__ import annotations

import cairosvg


def convert_svg_to_png(svg_path):
    png_path = svg_path.replace(".svg", ".png")
    if not os.path.exists(png_path):  # Avoid reprocessing
        try:
            cairosvg.svg2png(url=svg_path, write_to=png_path)
            print(f"Converted {svg_path} to PNG.")
        except Exception as e:
            print(f"Failed to convert {svg_path} to PNG: {e}")
    return png_path


import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_subplots_for_bin(bin_name, fit_type):
    # Define the exact directory structure
    base_dir = f"{bin_name}_fits"
    data_dirs = {"DATA": os.path.join(base_dir, "DATA"), "MC": os.path.join(base_dir, "MC")}

    # Create output directory for combined plots
    output_dir = os.path.join(base_dir, "combined_plots")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)

    # Track if we found any files
    files_found = False

    # Define plot positions and titles
    plot_config = [
        {"data_type": "DATA", "category": "pass", "position": (0, 0), "title": "DATA Pass"},
        {"data_type": "DATA", "category": "fail", "position": (0, 1), "title": "DATA Fail"},
        {"data_type": "MC", "category": "pass", "position": (1, 0), "title": "MC Pass"},
        {"data_type": "MC", "category": "fail", "position": (1, 1), "title": "MC Fail"},
    ]

    for config in plot_config:
        data_type = config["data_type"]
        category = config["category"]
        row, col = config["position"]
        title = config["title"]

        # Construct filename suffix
        pt_range = get_pt_range(bin_name)  # e.g., "35.00-40.00 GeV"
        pt_tag = pt_range.replace(".", "p").replace("-", "To").replace(" GeV", "")  # -> "35p00To40p00"

        hist_suffix = f"pt_{pt_tag}_{category.capitalize()}"  # e.g., "pt_35p00To40p00_Pass"

        # FIXED filename pattern to include 'barrel_1'
        filename = f"{data_type}_barrel_1_{fit_type}_fit_{bin_name}_{hist_suffix}.svg"
        filepath = os.path.join(data_dirs[data_type], filename)

        ax = fig.add_subplot(gs[row, col])

        if os.path.exists(filepath):
            files_found = True
            try:
                png_path = convert_svg_to_png(filepath)
                img = mpimg.imread(png_path)
                ax.imshow(img)
                ax.set_title(title, fontsize=25, pad=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading:\n{os.path.basename(filepath)}", ha="center", va="center", fontsize=10)
                ax.set_title(f"{title} (corrupted)", fontsize=14, pad=10)
                print(f"Warning: Could not read {filepath} - {e!s}")
        else:
            ax.text(0.5, 0.5, f"File not found:\n{filename}", ha="center", va="center", fontsize=10)
            ax.set_title(f"{title} (missing)", fontsize=14, pad=10)
            print(f"Warning: File not found - {filepath}")

        ax.axis("off")

    if files_found:
        # Add overall title with bin info
        bin_number = bin_name.replace("bin", "")
        pt_range = get_pt_range(bin_name)
        fig.suptitle(f"Bin {bin_number} ({pt_range})\nFit Type: {fit_type.replace('_', ' ').title()}", fontsize=18, y=0.98)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save high-quality PNG
        output_filename = os.path.join(output_dir, f"{bin_name}_{fit_type}_combined.png")
        plt.savefig(output_filename, bbox_inches="tight", dpi=150)
        print(f"Successfully created combined plot:\n{output_filename}")
    else:
        print(f"No valid plot files found for {bin_name} with fit type {fit_type}")

    plt.close()


def get_pt_range(bin_name):
    pt_ranges = {
        "bin00": "5.00-8.00 GeV",
        "bin01": "8.00-10.00 GeV",
        "bin02": "10.00-15.00 GeV",
        "bin03": "15.00-20.00 GeV",
        "bin04": "20.00-30.00 GeV",
        "bin05": "30.00-35.00 GeV",
        "bin06": "35.00-40.00 GeV",
        "bin07": "40.00-45.00 GeV",
        "bin08": "45.00-50.00 GeV",
        "bin09": "50.00-55.00 GeV",
        "bin10": "55.00-60.00 GeV",
        "bin11": "60.00-80.00 GeV",
        "bin12": "80.00-100.00 GeV",
        "bin13": "100.00-150.00 GeV",
        "bin14": "150.00-250.00 GeV",
        "bin15": "250.00-400.00 GeV",
    }
    return pt_ranges.get(bin_name, "Unknown Range")


if __name__ == "__main__":
    # Available options
    available_bins = [f"bin{str(i).zfill(2)}" for i in range(16)]
    available_fit_types = ["dcb_ps", "dcb_lin", "dcb_exp", "dcb_cheb", "dv_ps", "dv_lin", "dv_exp", "dv_cheb", "dg_ps", "dg_lin", "dg_exp", "dg_cheb"]

    # Set up argument parser with detailed help
    parser = argparse.ArgumentParser(
        description="Combine Z mass fit plots into 2x2 grid (DATA/MC × Pass/Fail)", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bin", type=str, required=True, choices=available_bins, help="Which pt bin to process (e.g., bin00, bin15)")
    parser.add_argument("--type", type=str, required=True, choices=available_fit_types, help="Which fit type to combine")
    parser.add_argument("--output-dir", type=str, default="combined_plots", help="Subdirectory name for output plots (created within bin folder)")

    args = parser.parse_args()

    print(f"\nProcessing {args.bin} with fit type {args.type}...")
    create_subplots_for_bin(args.bin, args.type)
