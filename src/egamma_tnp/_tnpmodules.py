import dask_awkward as dak
from coffea.lumi_tools import LumiMask


def apply_lumimasking(events, goldenjson):
    lumimask = LumiMask(goldenjson)
    mask = lumimask(events.run, events.luminosityBlock)
    return events[mask]


def filter_events(events, pt):
    events = events[dak.num(events.Electron) >= 2]
    abs_eta = abs(events.Electron.eta)
    pass_eta_ebeegap = (abs_eta < 1.4442) | (abs_eta > 1.566)
    pass_tight_id = events.Electron.cutBased == 4
    pass_pt = events.Electron.pt > pt
    pass_eta = abs_eta <= 2.5
    pass_selection = pass_pt & pass_eta & pass_eta_ebeegap & pass_tight_id
    n_of_tags = dak.sum(pass_selection, axis=1)
    good_events = events[n_of_tags >= 2]
    good_locations = pass_selection[n_of_tags >= 2]

    return good_events, good_locations


def trigger_match(electrons, trigobjs, pt):
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == 11
    pass_wptight = trigobjs.filterBits & (0x1 << 1) == 2
    trigger_cands = trigobjs[pass_pt & pass_id & pass_wptight]

    delta_r = electrons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = dak.sum(dak.sum(pass_delta_r, axis=1), axis=1)
    trig_matched_locs = n_of_trigger_matches >= 1

    return trig_matched_locs


def find_probes(tags, probes, trigobjs, pt):
    trig_matched_tag = trigger_match(tags, trigobjs, pt)
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
    trig_matched_probe = trigger_match(all_probes, trigobjs, pt)
    passing_probes = all_probes[trig_matched_probe]

    return passing_probes, all_probes


def perform_tnp(events, pt, goldenjson, extra_filter, extra_filter_args):
    if extra_filter is not None:
        events = extra_filter(events, **extra_filter_args)
    if goldenjson is not None:
        events = apply_lumimasking(events, goldenjson)
    good_events, good_locations = filter_events(events, pt)
    ele_for_tnp = good_events.Electron[good_locations]

    zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])
    zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])
    p1, a1 = find_probes(zcands1.tag, zcands1.probe, good_events.TrigObj, pt)
    p2, a2 = find_probes(zcands2.tag, zcands2.probe, good_events.TrigObj, pt)

    return p1, a1, p2, a2


def get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args):
    p1, a1, p2, a2 = perform_tnp(
        events, pt, goldenjson, extra_filter, extra_filter_args
    )

    pt_pass1 = dak.flatten(p1.pt)
    pt_pass2 = dak.flatten(p2.pt)
    pt_all1 = dak.flatten(a1.pt)
    pt_all2 = dak.flatten(a2.pt)

    eta_pass1 = dak.flatten(p1.eta)
    eta_pass2 = dak.flatten(p2.eta)
    eta_all1 = dak.flatten(a1.eta)
    eta_all2 = dak.flatten(a2.eta)

    phi_pass1 = dak.flatten(p1.phi)
    phi_pass2 = dak.flatten(p2.phi)
    phi_all1 = dak.flatten(a1.phi)
    phi_all2 = dak.flatten(a2.phi)

    return (
        pt_pass1,
        pt_pass2,
        pt_all1,
        pt_all2,
        eta_pass1,
        eta_pass2,
        eta_all1,
        eta_all2,
        phi_pass1,
        phi_pass2,
        phi_all1,
        phi_all2,
    )


def get_and_compute_arrays(
    events, pt, goldenjson, scheduler, progress, extra_filter, extra_filter_args
):
    import dask
    from dask.diagnostics import ProgressBar

    (
        pt_pass1,
        pt_pass2,
        pt_all1,
        pt_all2,
        eta_pass1,
        eta_pass2,
        eta_all1,
        eta_all2,
        phi_pass1,
        phi_pass2,
        phi_all1,
        phi_all2,
    ) = get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res = dask.compute(
        pt_pass1,
        pt_pass2,
        pt_all1,
        pt_all2,
        eta_pass1,
        eta_pass2,
        eta_all1,
        eta_all2,
        phi_pass1,
        phi_pass2,
        phi_all1,
        phi_all2,
        scheduler=scheduler,
    )

    if progress:
        pbar.unregister()

    return res


def get_tnp_histograms(events, pt, goldenjson, extra_filter, extra_filter_args):
    import json
    import os

    import hist
    from hist.dask import Hist

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    ptbins = config["ptbins"]
    etabins = config["etabins"]
    phibins = config["phibins"]

    (
        pt_pass1,
        pt_pass2,
        pt_all1,
        pt_all2,
        eta_pass1,
        eta_pass2,
        eta_all1,
        eta_all2,
        phi_pass1,
        phi_pass2,
        phi_all1,
        phi_all2,
    ) = get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args)

    ptaxis = hist.axis.Variable(ptbins, name="pt")
    hpt_all = Hist(ptaxis)
    hpt_pass = Hist(ptaxis)

    etaaxis = hist.axis.Variable(etabins, name="eta")
    heta_all = Hist(etaaxis)
    heta_pass = Hist(etaaxis)

    phiaxis = hist.axis.Variable(phibins, name="phi")
    hphi_all = Hist(phiaxis)
    hphi_pass = Hist(phiaxis)

    hpt_pass.fill(pt_pass1)
    hpt_pass.fill(pt_pass2)
    hpt_all.fill(pt_all1)
    hpt_all.fill(pt_all2)
    heta_pass.fill(eta_pass1)
    heta_pass.fill(eta_pass2)
    heta_all.fill(eta_all1)
    heta_all.fill(eta_all2)
    hphi_pass.fill(phi_pass1)
    hphi_pass.fill(phi_pass2)
    hphi_all.fill(phi_all1)
    hphi_all.fill(phi_all2)

    return hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all


def get_and_compute_tnp_histograms(
    events, pt, goldenjson, scheduler, progress, extra_filter, extra_filter_args
):
    import dask
    from dask.diagnostics import ProgressBar

    (
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
        hphi_pass,
        hphi_all,
    ) = get_tnp_histograms(events, pt, goldenjson, extra_filter, extra_filter_args)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res = dask.compute(
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
        hphi_pass,
        hphi_all,
        scheduler=scheduler,
    )

    if progress:
        pbar.unregister()

    return res
