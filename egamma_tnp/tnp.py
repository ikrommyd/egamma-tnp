import awkward as ak
import dask_awkward as dak
import hist
import numpy as np
from coffea.lumi_tools import LumiMask
from hist.dask import Hist


class DaskLumiMask:
    def __init__(self, lumimask):
        self._lumimask = lumimask

    def __call__(self, runs, lumis):
        out = self._lumimask(
            ak.typetracer.length_zero_if_typetracer(runs).to_numpy(),
            ak.typetracer.length_zero_if_typetracer(lumis).to_numpy(),
        )
        out = ak.Array(out)
        if ak.backend(runs, lumis) == "typetracer":
            out = ak.Array(
                out.layout.to_typetracer(forget_length=True), behavior=out.behavior
            )
        return out


def lumimask(events, jsonfile):
    eager_lumimask = LumiMask(jsonfile)
    dask_lumimask = DaskLumiMask(eager_lumimask)
    return dak.map_partitions(dask_lumimask, events.run, events.luminosityBlock)


def apply_lumimasking(events, goldenjson):
    mask = lumimask(events, goldenjson)
    return events[mask]


def filter_events(events):
    events = events[dak.num(events.Electron) >= 2]
    abs_eta = abs(events.Electron.eta)
    pass_eta_ebeegap = (abs_eta < 1.4442) | (abs_eta > 1.566)
    pass_tight_id = events.Electron.cutBased == 4
    pass_pt = events.Electron.pt > 31
    pass_eta = abs_eta <= 2.5
    pass_selection = pass_pt & pass_eta & pass_eta_ebeegap & pass_tight_id
    n_of_tags = dak.sum(pass_selection, axis=1)
    good_events = events[n_of_tags >= 2]
    good_locations = pass_selection[n_of_tags >= 2]

    return good_events, good_locations


def trigger_match(electrons, trigobjs):
    pass_pt = trigobjs.pt > 31
    pass_id = abs(trigobjs.id) == 11
    pass_wptight = trigobjs.filterBits & (0x1 << 1) == 2
    trigger_cands = trigobjs[pass_pt & pass_id & pass_wptight]

    delta_r = electrons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = dak.sum(dak.sum(pass_delta_r, axis=1), axis=1)
    trig_matched_locs = n_of_trigger_matches >= 1

    return trig_matched_locs


def find_probes(tags, probes, trigobjs):
    trig_matched_tag = trigger_match(tags, trigobjs)
    tags = tags[trig_matched_tag]
    probes = probes[trig_matched_tag]
    trigobjs = trigobjs[trig_matched_tag]

    dr = tags.delta_r(probes)
    mass = (tags + probes).mass

    in_mass_window = abs(mass - 91.1876) < 30
    opposite_charge = tags.charge * probes.charge == -1

    isZ = in_mass_window & opposite_charge
    dr_condition = dr > 0.0

    all_probes = probes[isZ & dr_condition]
    trig_matched_probe = trigger_match(all_probes, trigobjs)
    passing_probes = all_probes[trig_matched_probe]

    return passing_probes, all_probes


def perform_tnp(events, goldenjson):
    events = apply_lumimasking(events, goldenjson)
    good_events, good_locations = filter_events(events)
    ele_for_tnp = good_events.Electron[good_locations]

    zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])
    zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])
    p1, a1 = find_probes(zcands1.tag, zcands1.probe, good_events.TrigObj)
    p2, a2 = find_probes(zcands2.tag, zcands2.probe, good_events.TrigObj)

    return p1, a1, p2, a2


def get_tnp_histograms(events, goldenjson):
    from .config import etabins, ptbins

    p1, a1, p2, a2 = perform_tnp(events, goldenjson)

    ptaxis = hist.axis.Variable(ptbins, name="pt")
    hpt_all = Hist(ptaxis)
    hpt_pass = Hist(ptaxis)

    etaaxis = hist.axis.Variable(etabins, name="eta")
    heta_all = Hist(etaaxis)
    heta_pass = Hist(etaaxis)

    absetaaxis = hist.axis.Variable(np.unique(np.abs(etabins)), name="abseta")
    habseta_all = Hist(absetaaxis)
    habseta_pass = Hist(absetaaxis)

    # Fill for p1, a1
    hpt_all.fill(dak.flatten(a1.pt))
    hpt_pass.fill(dak.flatten(p1.pt))

    heta_all.fill(dak.flatten(a1.eta))
    heta_pass.fill(dak.flatten(p1.eta))

    habseta_all.fill(abs(dak.flatten(a1.eta)))
    habseta_pass.fill(abs(dak.flatten(p1.eta)))

    # Fill for p2, a2
    hpt_all.fill(dak.flatten(a2.pt))
    hpt_pass.fill(dak.flatten(p2.pt))

    heta_all.fill(dak.flatten(a2.eta))
    heta_pass.fill(dak.flatten(p2.eta))

    habseta_all.fill(abs(dak.flatten(a2.eta)))
    habseta_pass.fill(abs(dak.flatten(p2.eta)))

    return hpt_all, hpt_pass, heta_all, heta_pass, habseta_all, habseta_pass


def get_and_compute_tnp_histograms(
    events, goldenjson, scheduler="threads", progress=True
):
    import dask
    from dask.diagnostics import ProgressBar

    (
        hpt_all,
        hpt_pass,
        heta_all,
        heta_pass,
        habseta_all,
        habseta_pass,
    ) = get_tnp_histograms(events, goldenjson)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res = dask.compute(
        hpt_all,
        hpt_pass,
        heta_all,
        heta_pass,
        habseta_all,
        habseta_pass,
        scheduler=scheduler,
    )

    if progress:
        pbar.unregister()

    return res
