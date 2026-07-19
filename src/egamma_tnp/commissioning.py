from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import warnings

import awkward as ak
import dask
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from coffea.lumi_tools import LumiMask

from egamma_tnp.utils.pileup import create_correction, get_pileup_weight, load_correction

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")


class Commissioning_hists:
    def __init__(
        self,
        fileset_json,
        var_json,
        tag_pt_cut=35,
        probe_pt_cut=10,
        tag_max_SC_abseta=1.4442,
        probe_max_SC_abseta=2.5,
        Z_mass_range=[80, 100],
        cutbased_id="cutBased >= 2",
        hist_full_range=False,
        use_dask=True,
        tag_sc_abseta_range=None,
        probe_abseta_range=None,
        extra_tags_mask=None,
        extra_probes_mask=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        self.fileset_json = fileset_json
        self.var_json = var_json
        self.tag_pt_cut = tag_pt_cut
        self.probe_pt_cut = probe_pt_cut
        self.tag_max_SC_abseta = tag_max_SC_abseta
        self.probe_max_SC_abseta = probe_max_SC_abseta
        self.Z_mass_range = Z_mass_range
        self.cutbased_id = cutbased_id
        self.hist_full_range = hist_full_range
        self.use_dask = use_dask
        self.tag_sc_abseta_range = tag_sc_abseta_range
        self.probe_abseta_range = probe_abseta_range
        self.extra_tags_mask = extra_tags_mask
        self.extra_probes_mask = extra_probes_mask
        self.extra_filter = extra_filter
        self.extra_filter_args = extra_filter_args

        self.data_color_list = ["black", "saddlebrown", "green"]
        self.data_marker_list = ["o", "^", "v"]
        self.MC_color_list = ["dodgerblue", "limegreen", "m"]

        self.MC2024_pileup = ak.Array(
            [
                0.000001113213989874036,
                0.0000015468091923934005,
                0.0000021354450190071103,
                0.0000029291338469218304,
                0.000003992042749954428,
                0.000005405833377829933,
                0.0000072735834977346375,
                0.000009724331962846662,
                0.000012918274216272392,
                0.0000170526149622419,
                0.000022368057920838783,
                0.000029155879571161017,
                0.000037765494920067983,
                0.000048612379683335426,
                0.00006218616667282416,
                0.0000790586873656787,
                0.00009989168626752147,
                0.00012544390044306957,
                0.00015657717511638106,
                0.0001942612850563351,
                0.00023957715777935895,
                0.0002937182560760011,
                0.0003579899817243151,
                0.00043380711678651587,
                0.000522689529793492,
                0.0006262566455876264,
                0.0007462215105090346,
                0.0008843856747819444,
                0.0010426365496294917,
                0.0012229493551649353,
                0.0014273962186958265,
                0.0016581653538683676,
                0.0019175934660088899,
                0.002208214475733204,
                0.0025328271885712,
                0.002894583494129698,
                0.0032970968725650703,
                0.003744568249914057,
                0.0042419224608759715,
                0.004794943739086138,
                0.005410392925019933,
                0.006096082865554651,
                0.0068608824777247265,
                0.007714615180320339,
                0.00866781516074428,
                0.009731306710317893,
                0.010915579065169637,
                0.012229942947193852,
                0.013681475759704227,
                0.01527378959087405,
                0.017005687963005053,
                0.01886981038809762,
                0.020851393740706936,
                0.022927300903420315,
                0.02506547465186719,
                0.027224963763031827,
                0.029356636191662466,
                0.03140464101794246,
                0.033308610261160686,
                0.0350065105276433,
                0.03643797256026551,
                0.03754785536879911,
                0.038289751851088086,
                0.03862912367701764,
                0.038545769885154825,
                0.0380353863028649,
                0.037110056331124616,
                0.03579761783198069,
                0.03413996250306416,
                0.032190428693587354,
                0.030010532709785615,
                0.027666337121796485,
                0.02522477201906586,
                0.02275020654952582,
                0.0203015184549408,
                0.017929837607910185,
                0.01567705686218363,
                0.013575121200719511,
                0.011646034074885066,
                0.009902465083581686,
                0.00834880944787664,
                0.006982537399290253,
                0.005795678071150336,
                0.004776303267056856,
                0.003909906169865526,
                0.0031806032900314855,
                0.0025721201685428422,
                0.002068549223952546,
                0.0016548897331626839,
                0.001317394620618747,
                0.0010437568832815229,
                0.0008231711610503282,
                0.0006463045706591073,
                0.0005052068974853441,
                0.0003931848671938052,
                0.00030465950668724277,
                0.00023502024949971086,
                0.00018048485172410123,
                0.00013797053104661156,
                0.00010497902395277662,
            ]
        )

        if self.use_dask:
            from hist.dask import Hist
        else:
            from hist import Hist

        self.Hist = Hist

    def load_json(self, json_path):
        with open(json_path) as file:
            fileset_json = json.load(file)

        return fileset_json

    def save_json(self, content, json_path):
        with open(json_path, "w") as file:
            json.dump(content, file, indent=4)

    def load_samples(self, samples_config):
        num_reference = 0
        reference_sample_dict = {}
        target_samples_dict = {}
        common_fileds = []

        print("\n")

        for key in samples_config.keys():
            if samples_config[key]["is_reference"]:
                if num_reference > 0:
                    print("There are more than one reference samples. Exiting now...")
                    sys.exit()

                num_reference += 1
                print(f"\t INFO: Loading {key} parquet files")
                reference_sample_dict[key] = self.basic_preselection(samples_config[key])
                common_fileds.append(set(reference_sample_dict[key].fields))

            else:
                print(f"\t INFO: Loading {key} parquet files")
                target_samples_dict[key] = self.basic_preselection(samples_config[key])
                common_fileds.append(set(target_samples_dict[key].fields))

        common_fileds = set.intersection(*common_fileds)

        return reference_sample_dict, target_samples_dict, common_fileds

    def fetch_golden_json(self, goldenjson_url):
        # Local path to save the JSON file
        split_list = goldenjson_url.split("/")
        local_file_dir = "/".join(split_list[-3:-1])
        local_file_path = "/".join(split_list[-3:])

        if not os.path.exists(local_file_path):
            print(f"\t INFO: Fetching goldenjson from {goldenjson_url}")
            os.makedirs(local_file_dir, exist_ok=True)

            try:
                # Fetch the JSON data from the URL
                with urllib.request.urlopen(goldenjson_url) as response:
                    data = response.read().decode("utf-8")

                # Parse the JSON data
                json_data = json.loads(data)

                # Save the JSON data to a local file
                with open(local_file_path, "w") as file:
                    json.dump(json_data, file, indent=4)

                print(f"\t JSON data saved to {local_file_path} \n")

            except urllib.error.URLError as e:
                print(f"Error fetching data from {goldenjson_url}: {e}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON data: {e}")
            except OSError as e:
                print(f"Error saving JSON to file: {e}")

        return local_file_path

    def get_marker_index(self, data_idx, MC_idx, samples_config):
        if samples_config["isMC"]:
            MC_idx += 1
            marker_style = [self.MC_color_list[MC_idx], None]
        else:
            data_idx += 1
            marker_style = [self.data_color_list[data_idx], self.data_marker_list[data_idx]]

        return data_idx, MC_idx, marker_style

    def basic_preselection(self, sample_config):
        events = ak.from_parquet(sample_config["files"])

        if not sample_config["isMC"]:
            if "https" in sample_config["goldenjson"]:  # load and save golden json, if from a url
                sample_config["goldenjson"] = self.fetch_golden_json(sample_config["goldenjson"])

            lumimask = LumiMask(sample_config["goldenjson"])
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        pass_loose_cutBased = events[self.cutbased_id]

        pass_tag_pt_cut = events.tag_Ele_pt > self.tag_pt_cut
        pass_probe_pt_cut = events.el_pt > self.probe_pt_cut

        pass_tag_SC_abseta = abs(events.tag_Ele_eta + events.tag_Ele_deltaEtaSC) < self.tag_max_SC_abseta
        pass_probe_SC_abseta = abs(events.el_eta + events.el_deltaEtaSC) < self.probe_max_SC_abseta

        is_in_mass_range = (events.pair_mass > self.Z_mass_range[0]) & (events.pair_mass < self.Z_mass_range[1])

        events = events[pass_loose_cutBased & pass_tag_pt_cut & pass_probe_pt_cut & pass_tag_SC_abseta & pass_probe_SC_abseta & is_in_mass_range]

        return events

    def get_histogram_with_overflow(self, var_config, var_values, probe_region, weight):
        if var_config["custom_bin_edges"] is not None:
            bins = var_config["custom_bin_edges"]
        elif self.hist_full_range:
            bins = np.linspace(min(var_values), max(var_values), var_config["n_bins"]).tolist()
        else:
            hist_range = var_config["hist_range"] if isinstance(var_config["hist_range"], list) else var_config["hist_range"][probe_region]
            bins = np.linspace(hist_range[0], hist_range[1], var_config["n_bins"]).tolist()

        x_label = f"{var_config['xlabel']} ({probe_region})"

        h = self.Hist(
            hist.axis.Variable(bins, label=x_label, overflow=True, underflow=True),
            storage=hist.storage.Weight(),
        )
        h.fill(var_values, weight=weight)

        h_weighted_mean = hist.accumulators.WeightedMean().fill(var_values, weight=weight)

        return h, h_weighted_mean, h.sum().value

    def get_histograms_ratio(self, hist_target, hist_reference, yerr_target):
        ratio_values = hist_target.values() / hist_reference.values()
        ratio = self.Hist(self.Hist(*hist_target.axes))
        ratio[:] = np.nan_to_num(ratio_values)

        ratio_err_target = yerr_target / hist_target.values()
        ratio_err_target = np.nan_to_num(ratio_err_target)

        return ratio, ratio_err_target

    def plot_var_histogram(self, samples_config, vars_config, samples, var, pileup_corr, ax, probe_region, marker_style, norm_val=None):
        if samples_config["isMC"] and pileup_corr is not None:
            pileup_weight = get_pileup_weight(samples.Pileup_nTrueInt, pileup_corr)

        else:
            pileup_weight = 1

        h, h_weighted_mean, _ = self.get_histogram_with_overflow(vars_config[var], samples[var], probe_region=probe_region, weight=pileup_weight)

        legend = samples_config["Legend"]

        if "mass" in var:
            legend += f" [$\mu$={h_weighted_mean.value:.2f}, $\sigma$={np.sqrt(h_weighted_mean.variance):.2f}]"

        if norm_val is not None:
            h = h * (norm_val / h.sum().value)

        if samples_config["isMC"]:
            hist_type = "step"
            yerr = np.nan_to_num(np.sqrt(h.variances()))
            hep.histplot(h, label=legend, histtype=hist_type, yerr=yerr, flow="sum", ax=ax[0], color=marker_style[0])
        else:
            hist_type = "errorbar"
            yerr = np.nan_to_num(np.sqrt(h.values()))
            hep.histplot(h, label=legend, histtype=hist_type, yerr=yerr, flow="sum", ax=ax[0], color=marker_style[0], marker=marker_style[1], markersize=5)

        # hep.histplot(h, label=legend, histtype=hist_type, yerr=yerr, flow="sum", ax=ax[0], color=marker_style[0], marker=marker_style[1])

        bin_width = h.axes[0].widths[0]

        return h, yerr, bin_width

    def plot_ratio(self, samples_config, hist_target, hist_reference, yerr_target, ax, marker_style):
        ratio, ratio_error = self.get_histograms_ratio(hist_target, hist_reference, yerr_target)
        # hist_type = "band" if samples_config["isMC"] else "errorbar"

        if samples_config["isMC"]:
            hist_type = "step"
            hep.histplot(ratio, histtype=hist_type, flow="sum", ax=ax[1], color=marker_style[0], yerr=ratio_error)
        else:
            hist_type = "errorbar"
            hep.histplot(ratio, histtype=hist_type, flow="sum", ax=ax[1], color=marker_style[0], marker=marker_style[1], yerr=ratio_error, markersize=5)

        return 0

    def plot_fill_between_uncertainty(self, h, ax, marker_style):
        errors_den = np.sqrt(h.variances(flow=True)) / h.values(flow=True)

        errors_den = np.nan_to_num(errors_den)

        lower_bound = 1 - errors_den
        upper_bound = 1 + errors_den

        ax[1].fill_between(
            h.to_numpy(flow=True)[1][:-1], lower_bound, upper_bound, hatch="XXXXX", step="post", facecolor="none", edgecolor=marker_style[0], linewidth=0
        )
        return 0

    def create_and_save_histograms(self):
        vars_config = self.load_json(self.var_json)
        input_config = self.load_json(self.fileset_json)
        out_dir = input_config["output_dir"]

        # create directory to save the histograms
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            sys.exit(f"\t The directory '{out_dir}' already exists. Exiting now to avoid overwriting existing plots.")

        # save the input configuration in the plots directory, for future reference
        self.save_json(input_config, f"{out_dir}/input_config.json")

        input_fileset = input_config["samples_to_compare"]
        pileup_histogram = input_config["pileup_histogram"]

        reference_sample, target_samples, common_fileds = self.load_samples(input_fileset)

        # create pileup correction
        if (pileup_histogram is not None) and (not os.path.exists(f"{pileup_histogram.split('.')[0]}_correction_MC2024.json")):
            create_correction(
                pileup_histogram, self.MC2024_pileup, outfile=f"{pileup_histogram.split('.')[0]}_correction_MC2024.json", normalize_pu_mc_array=True
            )

        pileup_corr = load_correction(f"{pileup_histogram.split('.')[0]}_correction_MC2024.json")

        plt.style.use(hep.style.CMS)

        (reference_key,) = reference_sample

        for probe_region in ["EB", "EE"]:
            reference_sample_ = reference_sample[reference_key]

            data_marker_idx = -1
            MC_marker_idx = -1

            target_samples_ = {}
            target_samples_marker = {}
            if probe_region == "EB":
                # reference_sample_ = reference_sample_[abs(reference_sample_.el_superclusterEta) < 1.4442]
                reference_sample_ = reference_sample_[abs(reference_sample_.el_eta + reference_sample_.el_deltaEtaSC) < 1.4442]
                data_marker_idx, MC_marker_idx, reference_sample_marker = self.get_marker_index(data_marker_idx, MC_marker_idx, input_fileset[reference_key])
                for sample in target_samples:
                    target_sample_ = target_samples[sample]
                    # target_samples_[sample] = target_sample_[abs(target_sample_.el_superclusterEta) < 1.4442]
                    target_samples_[sample] = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) < 1.4442]
                    data_marker_idx, MC_marker_idx, target_samples_marker[sample] = self.get_marker_index(data_marker_idx, MC_marker_idx, input_fileset[sample])
            else:
                # reference_sample_ = reference_sample_[abs(reference_sample_.el_superclusterEta) > 1.566]
                reference_sample_ = reference_sample_[abs(reference_sample_.el_eta + reference_sample_.el_deltaEtaSC) > 1.566]
                data_marker_idx, MC_marker_idx, reference_sample_marker = self.get_marker_index(data_marker_idx, MC_marker_idx, input_fileset[reference_key])
                for sample in target_samples:
                    target_sample_ = target_samples[sample]
                    # target_samples_[sample] = target_sample_[abs(target_sample_.el_superclusterEta) > 1.566]
                    target_samples_[sample] = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) > 1.566]
                    data_marker_idx, MC_marker_idx, target_samples_marker[sample] = self.get_marker_index(data_marker_idx, MC_marker_idx, input_fileset[sample])

            print("\n")
            for var in vars_config.keys():
                if var in common_fileds:
                    print(f"\t INFO: Saving histogram for '{var}' for probes in {probe_region}")

                    # create directory to save the histogram of the variable
                    os.makedirs(f"{out_dir}/Variable_{var}_{probe_region}")

                    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]}, sharex=True, figsize=(9, 9))  # , constrained_layout=True)
                    fig.subplots_adjust(hspace=0.07)
                    fig.subplots_adjust(left=0.2)
                    hep.cms.label(input_config["CMS_label"], data=True, com=input_config["com"], lumi=input_config["lumi"], ax=ax[0], fontsize=15)

                    # for normalization
                    norm_sample = reference_sample_ if input_config["norm_sample"] in reference_sample.keys() else target_samples_[input_config["norm_sample"]]
                    h_, h_w_mean_, norm_val = self.get_histogram_with_overflow(vars_config[var], norm_sample[var], probe_region=probe_region, weight=1)

                    hist_reference, yerr_reference, bin_width = self.plot_var_histogram(
                        input_fileset[reference_key],
                        vars_config,
                        reference_sample_,
                        var,
                        pileup_corr,
                        ax,
                        probe_region=probe_region,
                        norm_val=norm_val,
                        marker_style=reference_sample_marker,
                    )
                    _ = self.plot_fill_between_uncertainty(hist_reference, ax, marker_style=reference_sample_marker)

                    for sample in target_samples_:
                        target_sample_ = target_samples_[sample]
                        hist_target_, yerr_target_, bin_width_ = self.plot_var_histogram(
                            input_fileset[sample],
                            vars_config,
                            target_sample_,
                            var,
                            pileup_corr,
                            ax,
                            probe_region=probe_region,
                            norm_val=norm_val,
                            marker_style=target_samples_marker[sample],
                        )
                        _ = self.plot_ratio(input_fileset[sample], hist_target_, hist_reference, yerr_target_, ax, marker_style=target_samples_marker[sample])

                    # get hist range
                    hist_range = (
                        vars_config[var]["hist_range"] if isinstance(vars_config[var]["hist_range"], list) else vars_config[var]["hist_range"][probe_region]
                    )

                    # plot reference line at y=1 for ratio plot
                    ax[1].plot(hist_range, [1, 1], color="black", linestyle="--", linewidth=1)

                    ax[0].set_xlim(hist_range)
                    ax[0].set_ylim([0, ax[0].set_ylim()[1] * 1.4])
                    ax[0].set_xlabel("")

                    ylbl = "Events" if vars_config[var]["custom_bin_edges"] is not None else f"Events / {bin_width:.2g}"
                    ax[0].set_ylabel(ylbl)
                    ax[0].yaxis.get_offset_text().set_x(-0.09)
                    ax[0].yaxis.get_offset_text().set_y(1.15)

                    ax[0].legend(fontsize=15)

                    ax[1].set_xlim(hist_range)
                    ax[1].set_ylim([0, 2])
                    ax[1].set_ylabel("Ratio")
                    # ax[1].grid(axis='y', linestyle='--', linewidth=0.5)

                    # save the histogram
                    plt.savefig(f"{out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.png", dpi=300)  # , pad_inches=0.35)
                    plt.savefig(f"{out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.pdf", dpi=300)  # , pad_inches=0.35)

                    # also save the plot in log scale
                    ax[0].set_ylim([1, ax[0].set_ylim()[1] * 800])
                    ax[0].set_yscale("log")
                    plt.savefig(f"{out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.png", dpi=300)  # , pad_inches=0.35)
                    plt.savefig(f"{out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.pdf", dpi=300)  # , pad_inches=0.35)
                    plt.clf()

            plt.close()

        return 0

    def submit(self):
        if self.use_dask:
            from dask.diagnostics import ProgressBar
            from dask.distributed import Client

            with Client(n_workers=1, threads_per_worker=24) as _:
                to_compute = self.create_and_save_histograms()
                with ProgressBar():
                    dask.compute(to_compute)

        else:
            self.create_and_save_histograms()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Plot commissioning histograms")

    # Add arguments
    parser.add_argument("--fileset_json", type=str, required=True, help="Path to the fileset JSON configuration file. Sample file: config/fileset_2024F.json")
    parser.add_argument(
        "--var_binning_json",
        type=str,
        default="config/default_commissionig_binning.json",  # Provide your default file path here
        help="Path to the variable JSON file defining the variables to plot and binning.",
    )
    parser.add_argument("--use_dask", type=bool, default=False, help="Whether to use Dask for parallel processing. Default is False.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the class
    out = Commissioning_hists(fileset_json=args.fileset_json, var_json=args.var_binning_json, use_dask=args.use_dask)

    # Execute the workflow
    out.submit()
