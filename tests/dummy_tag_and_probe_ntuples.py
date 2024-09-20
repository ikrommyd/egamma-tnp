from __future__ import annotations

import awkward as ak


def mini_tag_and_probe_electrons(events):
    events["el_eta_to_use"] = events.el_sc_eta
    events["tag_Ele_eta_to_use"] = events.tag_sc_eta

    if True:
        pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_tags]
    if False:
        pass_eta_ebeegap_probes = (abs(events.el_eta_to_use) < 1.4442) | (abs(events.el_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_probes]

    pass_pt_tags = events.tag_Ele_pt > 35
    pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < 2.17
    pass_abseta_probes = abs(events.el_eta_to_use) < 2.5
    opposite_charge = events.tag_Ele_q * events.el_q == -1
    events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & opposite_charge]

    pass_pt_probes = events.el_pt > 27
    pass_cutbased_id = events.passingCutBasedLoose122XV1 == 1
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(events.pair_mass - 91.1876) < 30
    else:
        in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
    all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
    passing_locs = all_probe_events.passingCutBasedTight122XV1 == 1
    passing_probe_events = all_probe_events[passing_locs]
    failing_probe_events = all_probe_events[~passing_locs]

    vars = ["el_pt", "el_eta", "truePU", "tag_Ele_eta", "event", "run", "lumi"]
    if cut_and_count:
        passing_probes = ak.zip({var: passing_probe_events[var] for var in vars})
        failing_probes = ak.zip({var: failing_probe_events[var] for var in vars})
    else:
        p_arrays = {var: passing_probe_events[var] for var in vars}
        p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
        f_arrays = {var: failing_probe_events[var] for var in vars}
        f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
        passing_probes = ak.zip(p_arrays)
        failing_probes = ak.zip(f_arrays)

    return passing_probes, failing_probes


def mini_tag_and_probe_photons(events):
    events["ph_eta_to_use"] = events.ph_sc_eta
    events["tag_Ele_eta_to_use"] = events.tag_sc_eta

    if True:
        pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_tags]
    if False:
        pass_eta_ebeegap_probes = (abs(events.ph_eta_to_use) < 1.4442) | (abs(events.ph_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_probes]

    pass_pt_tags = events.tag_Ele_pt > 35
    pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < 2.17
    pass_abseta_probes = abs(events.ph_eta_to_use) < 2.5
    events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes]

    pass_pt_probes = events.ph_et > 27
    pass_cutbased_id = events.passingCutBasedLoose122XV1 == 1
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(events.pair_mass - 91.1876) < 30
    else:
        in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
    all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
    passing_locs = all_probe_events.passingCutBasedTight122XV1 == 1
    passing_probe_events = all_probe_events[passing_locs]
    failing_probe_events = all_probe_events[~passing_locs]

    vars = ["ph_et", "ph_eta", "truePU", "tag_Ele_eta", "event", "run", "lumi"]
    if cut_and_count:
        passing_probes = ak.zip({var: passing_probe_events[var] for var in vars})
        failing_probes = ak.zip({var: failing_probe_events[var] for var in vars})
    else:
        p_arrays = {var: passing_probe_events[var] for var in vars}
        p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
        f_arrays = {var: failing_probe_events[var] for var in vars}
        f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
        passing_probes = ak.zip(p_arrays)
        failing_probes = ak.zip(f_arrays)

    return passing_probes, failing_probes


def nano_tag_and_probe_electrons(events):
    events["el_eta_to_use"] = events.el_superclusterEta
    events["tag_Ele_eta_to_use"] = events.tag_Ele_superclusterEta

    if True:
        pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_tags]
    if False:
        pass_eta_ebeegap_probes = (abs(events.el_eta_to_use) < 1.4442) | (abs(events.el_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_probes]

    pass_pt_tags = events.tag_Ele_pt > 35
    pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < 2.17
    pass_abseta_probes = abs(events.el_eta_to_use) < 2.5
    opposite_charge = events.tag_Ele_charge * events.el_charge == -1
    events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & opposite_charge]

    pass_pt_probes = events.el_pt > 27
    pass_cutbased_id = events["cutBased >= 2"] == 1
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(events.pair_mass - 91.1876) < 30
    else:
        in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
    all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
    passing_locs = all_probe_events["cutBased >= 4"] == 1
    passing_probe_events = all_probe_events[passing_locs]
    failing_probe_events = all_probe_events[~passing_locs]

    vars = ["el_pt", "el_eta", "PV_npvs", "tag_Ele_eta", "event", "run", "luminosityBlock"]
    if cut_and_count:
        passing_probes = ak.zip({var: passing_probe_events[var] for var in vars})
        failing_probes = ak.zip({var: failing_probe_events[var] for var in vars})
    else:
        p_arrays = {var: passing_probe_events[var] for var in vars}
        p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
        f_arrays = {var: failing_probe_events[var] for var in vars}
        f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
        passing_probes = ak.zip(p_arrays)
        failing_probes = ak.zip(f_arrays)

    return passing_probes, failing_probes


def nano_tag_and_probe_photons(events):
    events["ph_eta_to_use"] = events.ph_superclusterEta
    events["tag_Ele_eta_to_use"] = events.tag_Ele_superclusterEta

    if True:
        pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_tags]
    if False:
        pass_eta_ebeegap_probes = (abs(events.ph_eta_to_use) < 1.4442) | (abs(events.ph_eta_to_use) > 1.566)
        events = events[pass_eta_ebeegap_probes]

    pass_pt_tags = events.tag_Ele_pt > 35
    pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < 2.17
    pass_abseta_probes = abs(events.ph_eta_to_use) < 2.5
    events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes]

    pass_pt_probes = events.ph_pt > 27
    pass_cutbased_id = events["cutBased >= 1"] == 1
    cut_and_count = False
    if cut_and_count:
        in_mass_window = abs(events.pair_mass - 91.1876) < 30
    else:
        in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
    all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
    passing_locs = all_probe_events["cutBased >= 3"] == 1
    passing_probe_events = all_probe_events[passing_locs]
    failing_probe_events = all_probe_events[~passing_locs]

    vars = ["ph_pt", "ph_eta", "PV_npvs", "tag_Ele_eta", "event", "run", "luminosityBlock"]
    if cut_and_count:
        passing_probes = ak.zip({var: passing_probe_events[var] for var in vars})
        failing_probes = ak.zip({var: failing_probe_events[var] for var in vars})
    else:
        p_arrays = {var: passing_probe_events[var] for var in vars}
        p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
        f_arrays = {var: failing_probe_events[var] for var in vars}
        f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
        passing_probes = ak.zip(p_arrays)
        failing_probes = ak.zip(f_arrays)

    return passing_probes, failing_probes
