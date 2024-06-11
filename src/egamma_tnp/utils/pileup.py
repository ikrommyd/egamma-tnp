from __future__ import annotations

import correctionlib
import correctionlib.convert
import hist
import numpy as np
import uproot
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper


def load_correction(correction_file, name=None):
    ceval = correctionlib.CorrectionSet.from_file(correction_file)
    if name is not None:
        corr = ceval[name]
    else:
        corr = ceval[next(iter(ceval.keys()))]

    return correctionlib_wrapper(corr)


def create_correction(pu_data_histogram, pu_mc_array, outfile=None, normalize_pu_mc_array=False):
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


def get_pileup_weight(true_pileup, pileup_corr):
    if len(pileup_corr._corr.inputs) == 2:
        return pileup_corr(true_pileup, "nominal")
    return pileup_corr(true_pileup)
