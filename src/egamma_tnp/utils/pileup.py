from __future__ import annotations

import correctionlib
import correctionlib.convert
import dask_awkward as dak
import hist
import numpy as np
import uproot
from coffea.analysis_tools import Weights
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper


def load_correction(correction_file, name=None):
    ceval = correctionlib.CorrectionSet.from_file(correction_file)
    if name is not None:
        corr = ceval[name]
    else:
        corr = ceval[next(iter(ceval.keys()))]

    return correctionlib_wrapper(corr)


def create_correction(pu_data_histogram, pu_mc_array, outfile=None, normalize_pu_mc_array=False):
    pu_mc_array[pu_mc_array == 0.0] = 1e-10
    pu_data = uproot.open(pu_data_histogram)["pileup"].to_hist().density()
    pu_mc = pu_mc_array / np.sum(pu_mc_array) if normalize_pu_mc_array else pu_mc_array
    assert len(pu_data) == len(pu_mc), "Data and MC pileup distributions have different lengths"
    sfhist = hist.Hist(hist.axis.Variable(np.arange(len(pu_data) + 1), label="pileup"), label="pileup", name="Pileup")
    sfhist[:] = pu_data / pu_mc

    clibcorr = correctionlib.convert.from_histogram(sfhist)
    clibcorr.description = "Pileup Reweighting"
    clibcorr.inputs[0].description = "Number of true interactions"
    clibcorr.data.flow = "clamp"
    clibcorr.output.name = "weight"
    clibcorr.output.description = "Event weight for pileup reweighting"
    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="Pileup corrections",
        corrections=[clibcorr],
    )

    if outfile is not None:
        if outfile.endswith(".json"):
            with open(outfile, "w") as fout:
                fout.write(cset.json(exclude_unset=True))
        elif outfile.endswith(".json.gz"):
            import gzip

            with gzip.open(outfile, "wt") as fout:
                fout.write(cset.json(exclude_unset=True))
        else:
            raise ValueError("Outfile should either be a .json or .json.gz file")

    ceval = cset.to_evaluator()
    corr = ceval["Pileup"]
    np.testing.assert_allclose(corr.evaluate(np.arange(len(pu_data), dtype=float)), (pu_data / pu_mc), err_msg="Pileup correction does not match input data")
    return correctionlib_wrapper(corr)


def get_pileup_weight(true_pileup, pileup_corr, syst=False):
    if len(pileup_corr._corr.inputs) == 2:
        if syst:
            return pileup_corr(true_pileup, "nominal"), pileup_corr(true_pileup, "up"), pileup_corr(true_pileup, "down")
        else:
            return pileup_corr(true_pileup, "nominal")
    return pileup_corr(true_pileup)


def apply_pileup_weights(dileptons, events, sum_genw_before_presel=1.0, syst=False):
    if events.metadata.get("isMC"):
        weights = Weights(size=None, storeIndividual=True)
        if "pileupJSON" in events.metadata:
            pileup_corr = load_correction(events.metadata["pileupJSON"])
        elif "pileupData" in events.metadata and "pileupMC" in events.metadata:
            pileup_corr = create_correction(events.metadata["pileupData"], events.metadata["pileupMC"])
        else:
            pileup_corr = None

        if "genWeight" in events.fields:
            weights.add("genWeight", events.genWeight)
        else:
            weights.add("genWeight", dak.ones_like(events.event))

        if pileup_corr is not None:
            pileup_weight_nom, pileup_weight_up, pileup_weight_down = get_pileup_weight(dileptons.nTrueInt, pileup_corr, syst=syst)
            weights.add("Pileup", pileup_weight_nom, pileup_weight_up if syst else None, pileup_weight_down if syst else None)

            dileptons["weight_central"] = pileup_weight_nom
            if syst:
                dileptons["weight_central_PileupUp"] = weights.partial_weight(include=["Pileup"], modifier="PileupUp")
                dileptons["weight_central_PileupDown"] = weights.partial_weight(include=["Pileup"], modifier="PileupDown")

            dileptons["weight"] = weights.partial_weight(include=["Pileup", "genWeight"]) / sum_genw_before_presel
            if syst:
                dileptons["weight_PileupUp"] = weights.partial_weight(include=["Pileup", "genWeight"], modifier="PileupUp") / sum_genw_before_presel
                dileptons["weight_PileupDown"] = weights.partial_weight(include=["Pileup", "genWeight"], modifier="PileupDown") / sum_genw_before_presel

        else:
            dileptons["weight_central"] = dak.ones_like(events.event)
            dileptons["weight"] = dileptons["genWeight"] / sum_genw_before_presel

    return dileptons
