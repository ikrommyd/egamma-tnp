from __future__ import annotations

import awkward as ak
from coffea.nanoevents.methods import nanoaod

from egamma_tnp.utils import custom_delta_r


def trigger_match(leptons, trigobjs, pdgid, pt, filterbit):
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == pdgid
    pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = leptons.metric_table(trigger_cands, metric=custom_delta_r)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1
    return trig_matched_locs


def tag_and_probe_electrons(events, is_id):
    events["Electron", "eta_to_use"] = events.Electron.eta + events.Electron.deltaEtaSC
    events["Electron", "phi_to_use"] = events.Electron.phi

    good_events = events[events.HLT.Ele30_WPTight_Gsf]

    ij = ak.argcartesian([good_events.Electron, good_events.Electron])
    is_not_diag = ij["0"] != ij["1"]
    i, j = ak.unzip(ij[is_not_diag])
    zcands = ak.zip({"tag": good_events.Electron[i], "probe": good_events.Electron[j]})

    if True:
        tags = zcands.tag
        pass_eta_ebeegap_tags = (abs(tags.eta_to_use) < 1.4442) | (abs(tags.eta_to_use) > 1.566)
        zcands = zcands[pass_eta_ebeegap_tags]
    if False:
        probes = zcands.probe
        pass_eta_ebeegap_probes = (abs(probes.eta_to_use) < 1.4442) | (abs(probes.eta_to_use) > 1.566)
        zcands = zcands[pass_eta_ebeegap_probes]

    pass_tight_id_tags = zcands.tag.cutBased >= 4
    if is_id:
        pass_cutbased_id_probes = (True,)
    else:
        pass_cutbased_id_probes = zcands.probe.cutBased >= 4
    zcands = zcands[pass_tight_id_tags & pass_cutbased_id_probes]

    trigobjs = good_events.TrigObj
    pt_cond_tags = zcands.tag.pt > 35
    eta_cond_tags = abs(zcands.tag.eta_to_use) < 2.17
    pt_cond_probes = zcands.probe.pt > 27
    eta_cond_probes = abs(zcands.probe.eta_to_use) < 2.5
    trig_matched_tag = trigger_match(zcands.tag, trigobjs, 11, 30, 1)
    zcands = zcands[trig_matched_tag & pt_cond_tags & pt_cond_probes & eta_cond_tags & eta_cond_probes]
    events_with_tags = ak.num(zcands.tag, axis=1) >= 1
    zcands = zcands[events_with_tags]
    trigobjs = trigobjs[events_with_tags]
    good_events = good_events[events_with_tags]
    tags = zcands.tag
    probes = zcands.probe
    dr = tags.delta_r(probes)
    mass = (tags + probes).mass
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(mass - 91.1876) < 30
    else:
        in_mass_window = (mass > 50) & (mass < 130)
    opposite_charge = tags.charge * probes.charge == -1
    isZ = in_mass_window & opposite_charge
    dr_condition = dr > 0.0
    zcands = zcands[isZ & dr_condition]
    if is_id:
        trig_matched_probe = zcands.probe.cutBased >= 4
        hlt_filter = None
    else:
        trig_matched_probe = trigger_match(zcands.probe, trigobjs, 11, 32, 1)
        hlt_filter = "Ele32_WPTight_Gsf"
    if hlt_filter is None:
        passing_pairs = zcands[trig_matched_probe]
        failing_pairs = zcands[~trig_matched_probe]
    else:
        passing_pairs = zcands[trig_matched_probe & getattr(good_events.HLT, hlt_filter)]
        failing_pairs = zcands[~(trig_matched_probe & getattr(good_events.HLT, hlt_filter))]
    has_passing_probe = ak.num(passing_pairs) >= 1
    has_failing_probe = ak.num(failing_pairs) >= 1
    passing_pairs = passing_pairs[has_passing_probe]
    failing_pairs = failing_pairs[has_failing_probe]
    passing_probe_events = good_events[has_passing_probe]
    failing_probe_events = good_events[has_failing_probe]
    passing_probe_events["el"] = passing_pairs.probe
    failing_probe_events["el"] = failing_pairs.probe
    passing_probe_events["tag_Ele"] = passing_pairs.tag
    failing_probe_events["tag_Ele"] = failing_pairs.tag
    passing_probe_events["pair_mass"] = (passing_probe_events["el"] + passing_probe_events["tag_Ele"]).mass
    failing_probe_events["pair_mass"] = (failing_probe_events["el"] + failing_probe_events["tag_Ele"]).mass

    passing_probe_dict = {}
    failing_probe_dict = {}
    vars = ["Electron_pt", "tag_Ele_eta", "el_pt", "el_eta", "MET_pt", "event", "run", "luminosityBlock"]
    for var in vars:
        if var.startswith("el_"):
            passing_probe_dict[var] = passing_probe_events["el", var.removeprefix("el_")]
            failing_probe_dict[var] = failing_probe_events["el", var.removeprefix("el_")]
        elif var.startswith("tag_Ele_"):
            passing_probe_dict[var] = passing_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
            failing_probe_dict[var] = failing_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
        else:
            split = var.split("_", 1)
            if len(split) == 2:
                passing_probe_dict[var] = passing_probe_events[split[0], split[1]]
                failing_probe_dict[var] = failing_probe_events[split[0], split[1]]
            else:
                passing_probe_dict[var] = passing_probe_events[var]
                failing_probe_dict[var] = failing_probe_events[var]
    if not cut_and_count:
        passing_probe_dict["pair_mass"] = passing_probe_events.pair_mass
        failing_probe_dict["pair_mass"] = failing_probe_events.pair_mass
    passing_probes = ak.zip(passing_probe_dict, depth_limit=1)
    failing_probes = ak.zip(failing_probe_dict, depth_limit=1)

    return passing_probes, failing_probes


def tag_and_probe_photons(events, start_from_diphotons, is_id):
    # TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
    photon_dict = {field: events.Photon[field] for field in events.Photon.fields} | {
        "mass": ak.zeros_like(events.Photon.pt),
        "charge": ak.zeros_like(events.Photon.pt),
    }

    events["Photon"] = ak.zip(photon_dict, with_name="Photon", behavior=nanoaod.behavior)

    events["Photon", "eta_to_use"] = events.Photon.eta
    events["Photon", "phi_to_use"] = events.Photon.phi
    events["Electron", "eta_to_use"] = events.Electron.eta
    events["Electron", "phi_to_use"] = events.Electron.phi
    events["Photon", "charge"] = 0.0 * events.Photon.pt

    good_events = events[events.HLT.Ele30_WPTight_Gsf]

    if start_from_diphotons:
        ij = ak.argcartesian([good_events.Photon, good_events.Photon])
        is_not_diag = ij["0"] != ij["1"]
        i, j = ak.unzip(ij[is_not_diag])
        zcands = ak.zip({"tag": good_events.Photon[i], "probe": good_events.Photon[j]})
        pass_tight_id_tags = zcands.tag.cutBased >= 3
    else:
        ij = ak.argcartesian({"tag": good_events.Electron, "probe": good_events.Photon})
        tnp = ak.cartesian({"tag": good_events.Electron, "probe": good_events.Photon})
        probe_is_not_tag = (tnp.probe.electronIdx != ij.tag) & (tnp.tag.delta_r(tnp.probe) > 0.1)
        zcands = tnp[probe_is_not_tag]
        pass_tight_id_tags = zcands.tag.cutBased >= 4

    if is_id:
        pass_cutbased_id_probes = True
    else:
        pass_cutbased_id_probes = zcands.probe.cutBased >= 3
    zcands = zcands[pass_tight_id_tags & pass_cutbased_id_probes]

    if True:
        tags = zcands.tag
        pass_eta_ebeegap_tags = (abs(tags.eta_to_use) < 1.4442) | (abs(tags.eta_to_use) > 1.566)
        zcands = zcands[pass_eta_ebeegap_tags]
    if False:
        probes = zcands.probe
        pass_eta_ebeegap_probes = (abs(probes.eta_to_use) < 1.4442) | (abs(probes.eta_to_use) > 1.566)
        zcands = zcands[pass_eta_ebeegap_probes]

    trigobjs = good_events.TrigObj
    pt_cond_tags = zcands.tag.pt > 35
    eta_cond_tags = abs(zcands.tag.eta_to_use) < 2.17
    pt_cond_probes = zcands.probe.pt > 27
    eta_cond_probes = abs(zcands.probe.eta_to_use) < 2.5
    if start_from_diphotons:
        has_matched_electron_tags = (zcands.tag.electronIdx != -1) & (zcands.tag.pixelSeed)
        trig_matched_tag = trigger_match(zcands.tag.matched_electron, trigobjs, 11, 30, 1)
    else:
        has_matched_electron_tags = True
        trig_matched_tag = trigger_match(zcands.tag, trigobjs, 11, 30, 1)
    zcands = zcands[has_matched_electron_tags & trig_matched_tag & pt_cond_tags & pt_cond_probes & eta_cond_tags & eta_cond_probes]
    events_with_tags = ak.num(zcands.tag, axis=1) >= 1
    zcands = zcands[events_with_tags]
    trigobjs = trigobjs[events_with_tags]
    good_events = good_events[events_with_tags]
    tags = zcands.tag
    probes = zcands.probe
    dr = tags.delta_r(probes)
    mass = (tags + probes).mass
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(mass - 91.1876) < 30
    else:
        in_mass_window = (mass > 50) & (mass < 130)
    opposite_charge = True
    isZ = in_mass_window & opposite_charge
    dr_condition = dr > 0.0
    zcands = zcands[isZ & dr_condition]
    if is_id:
        trig_matched_probe = zcands.probe.cutBased >= 3
        hlt_filter = None
    else:
        trig_matched_probe = trigger_match(zcands.probe, trigobjs, 11, 32, 1)
        hlt_filter = "Ele32_WPTight_Gsf"
    if hlt_filter is None:
        passing_pairs = zcands[trig_matched_probe]
        failing_pairs = zcands[~trig_matched_probe]
    else:
        passing_pairs = zcands[trig_matched_probe & getattr(good_events.HLT, hlt_filter)]
        failing_pairs = zcands[~(trig_matched_probe & getattr(good_events.HLT, hlt_filter))]
    has_passing_probe = ak.num(passing_pairs) >= 1
    has_failing_probe = ak.num(failing_pairs) >= 1
    passing_pairs = passing_pairs[has_passing_probe]
    failing_pairs = failing_pairs[has_failing_probe]
    passing_probe_events = good_events[has_passing_probe]
    failing_probe_events = good_events[has_failing_probe]
    passing_probe_events["ph"] = passing_pairs.probe
    failing_probe_events["ph"] = failing_pairs.probe
    passing_probe_events["tag_Ele"] = passing_pairs.tag
    failing_probe_events["tag_Ele"] = failing_pairs.tag
    passing_probe_events["pair_mass"] = (passing_probe_events["ph"] + passing_probe_events["tag_Ele"]).mass
    failing_probe_events["pair_mass"] = (failing_probe_events["ph"] + failing_probe_events["tag_Ele"]).mass

    passing_probe_dict = {}
    failing_probe_dict = {}
    vars = ["Photon_pt", "tag_Ele_eta", "ph_pt", "ph_eta", "MET_pt", "event", "run", "luminosityBlock"]
    for var in vars:
        if var.startswith("ph_"):
            passing_probe_dict[var] = passing_probe_events["ph", var.removeprefix("ph_")]
            failing_probe_dict[var] = failing_probe_events["ph", var.removeprefix("ph_")]
        elif var.startswith("tag_Ele_"):
            passing_probe_dict[var] = passing_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
            failing_probe_dict[var] = failing_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
        else:
            split = var.split("_", 1)
            if len(split) == 2:
                passing_probe_dict[var] = passing_probe_events[split[0], split[1]]
                failing_probe_dict[var] = failing_probe_events[split[0], split[1]]
            else:
                passing_probe_dict[var] = passing_probe_events[var]
                failing_probe_dict[var] = failing_probe_events[var]
    if not cut_and_count:
        passing_probe_dict["pair_mass"] = passing_probe_events.pair_mass
        failing_probe_dict["pair_mass"] = failing_probe_events.pair_mass
    passing_probes = ak.zip(passing_probe_dict, depth_limit=1)
    failing_probes = ak.zip(failing_probe_dict, depth_limit=1)

    return passing_probes, failing_probes
