import dask_awkward as dak
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoAODSchema

from egamma_tnp._base_tagnprobe import BaseTagNProbe


class TagNProbeFromNanoAOD(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        for_trigger,
        egm_nano=False,
        *,
        filter="None",
        trigger_pt=None,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        filterbit=None,
        cutbased_id=None,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        hlt_filter=None,
    ):
        """Tag and Probe efficiency from NanoAOD and EGamma NanoAOD.
        Can only perform trigger efficiencies at the moment.

        Parameters
        ----------
        fileset: dict
            The fileset to calculate the trigger efficiencies for.
        for_trigger: bool
            Whether the filter is a trigger or not.
        egm_nano: bool, optional
            Whether the input fileset is EGamma NanoAOD or NanoAOD. The default is False.
        filter: str
            The name of the filter to calculate the efficiencies for.
        trigger_pt: int or float, optional
            The Pt threshold of the trigger to calculate the efficiencies over that threshold.
            If None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
            The default is None.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 35.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
            Should be very slightly below the Pt threshold of the filter.
            If it is None, it will attempt to infer it from the filter name.
            If it fails to do so, it will set it to 0.
        tags_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the tag electrons. The default is 2.5.
        cutbased_id: int, optional
            The number of the cutbased ID to apply to the electrons.
            If None, no cutbased ID is applied. The default is None.
        goldenjson: str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
        use_sc_eta : bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi : bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        hlt_filter : str, optional
            The HLT filter to also require an event to have passed to consider a probe belonging to that event as passing.
            If None, no such requirement is applied. The default is None.
        """
        if for_trigger is False:
            raise NotImplementedError("Only trigger efficiencies are supported at the moment.")
        if use_sc_phi and not egm_nano:
            raise NotImplementedError("Supercluster Phi is only available for EGamma NanoAOD.")
        if for_trigger and filterbit is None:
            raise ValueError("TrigObj filerbit must be provided for trigger efficiencies.")
        if filter == "None" and trigger_pt is None and for_trigger:
            raise ValueError("An HLT filter name or a trigger Pt threshold must be provided for trigger efficiencies.")
        if trigger_pt is None:
            from egamma_tnp.utils.misc import find_pt_threshold

            self.trigger_pt = find_pt_threshold(filter)
        else:
            self.trigger_pt = trigger_pt
        super().__init__(
            fileset=fileset,
            filter=filter,
            tags_pt_cut=tags_pt_cut,
            probes_pt_cut=probes_pt_cut,
            tags_abseta_cut=tags_abseta_cut,
            cutbased_id=cutbased_id,
            goldenjson=goldenjson,
            extra_filter=extra_filter,
            extra_filter_args=extra_filter_args,
            use_sc_eta=use_sc_eta,
            use_sc_phi=use_sc_phi,
            avoid_ecal_transition_tags=avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            schemaclass=NanoAODSchema,
        )
        self.for_trigger = for_trigger
        self.egm_nano = egm_nano
        self.filterbit = filterbit
        self.hlt_filter = hlt_filter

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"TagNProbeFromNanoAOD({self.filter}, Number of files: {n_of_files}, Golden JSON: {self.goldenjson})"

    def _find_probes(self, events, cut_and_count, vars):
        if vars is None:
            vars = ["pt", "eta", "phi"]
        if self.use_sc_eta:
            if self.egm_nano:
                events["Electron", "eta"] = events.Electron.superclusterEta
            else:
                events["Electron", "eta"] = events.Electron.eta + events.Electron.deltaEtaSC
        if self.use_sc_phi:
            events["Electron", "phi"] = events.Electron.superclusterPhi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        good_events, good_locations = _filter_events(events, self.cutbased_id)
        ele_for_tnp = good_events.Electron[good_locations]
        zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])

        if self.avoid_ecal_transition_tags:
            tags1 = zcands1.tag
            pass_eta_ebeegap_tags1 = (abs(tags1.eta) < 1.4442) | (abs(tags1.eta) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_tags1]
        if self.avoid_ecal_transition_probes:
            probes1 = zcands1.probe
            pass_eta_ebeegap_probes1 = (abs(probes1.eta) < 1.4442) | (abs(probes1.eta) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_probes1]

        p1, f1 = _process_zcands(
            zcands=zcands1,
            good_events=good_events,
            trigger_pt=self.trigger_pt,
            pt_tags=self.tags_pt_cut,
            pt_probes=self.probes_pt_cut,
            abseta_tags=self.tags_abseta_cut,
            filterbit=self.filterbit,
            cut_and_count=cut_and_count,
            hlt_filter=self.hlt_filter,
        )

        if cut_and_count:
            zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])

            if self.avoid_ecal_transition_tags:
                tags2 = zcands2.tag
                pass_eta_ebeegap_tags2 = (abs(tags2.eta) < 1.4442) | (abs(tags2.eta) > 1.566)
                zcands2 = zcands2[pass_eta_ebeegap_tags2]
            if self.avoid_ecal_transition_probes:
                probes2 = zcands2.probe
                pass_eta_ebeegap_probes2 = (abs(probes2.eta) < 1.4442) | (abs(probes2.eta) > 1.566)
                zcands2 = zcands2[pass_eta_ebeegap_probes2]

            p2, f2 = _process_zcands(
                zcands=zcands2,
                good_events=good_events,
                trigger_pt=self.trigger_pt,
                pt_tags=self.tags_pt_cut,
                pt_probes=self.probes_pt_cut,
                abseta_tags=self.tags_abseta_cut,
                filterbit=self.filterbit,
                cut_and_count=cut_and_count,
                hlt_filter=self.hlt_filter,
            )

            p, f = dak.concatenate([p1, p2]), dak.concatenate([f1, f2])

        else:
            p, f = p1, f1

        if cut_and_count:
            passing_probes = dak.flatten(dak.zip({var: p[var] for var in vars}))
            failing_probes = dak.flatten(dak.zip({var: f[var] for var in vars}))
        else:
            p_arrays = {var: p[var] for var in vars}
            p_arrays["pair_mass"] = p["pair_mass"]
            f_arrays = {var: f[var] for var in vars}
            f_arrays["pair_mass"] = f["pair_mass"]
            passing_probes = dak.flatten(dak.zip(p_arrays))
            failing_probes = dak.flatten(dak.zip(f_arrays))

        return passing_probes, failing_probes


def _filter_events(events, cutbased_id):
    pass_hlt = events.HLT.Ele30_WPTight_Gsf
    two_electrons = dak.num(events.Electron) == 2
    abs_eta = abs(events.Electron.eta)
    pass_tight_id = events.Electron.cutBased == cutbased_id
    pass_eta = abs_eta <= 2.5
    pass_selection = pass_hlt & two_electrons & pass_eta & pass_tight_id
    n_of_tags = dak.sum(pass_selection, axis=1)
    good_events = events[n_of_tags == 2]
    good_locations = pass_selection[n_of_tags == 2]
    return good_events, good_locations


def _trigger_match(electrons, trigobjs, pt, filterbit):
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == 11
    pass_filterbit = trigobjs.filterBits & (0x1 << filterbit) > 0
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = electrons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = dak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1
    return trig_matched_locs


def _process_zcands(
    zcands,
    good_events,
    trigger_pt,
    pt_tags,
    pt_probes,
    abseta_tags,
    filterbit,
    cut_and_count,
    hlt_filter,
):
    trigobjs = good_events.TrigObj
    pt_cond_tags = zcands.tag.pt > pt_tags
    eta_cond_tags = abs(zcands.tag.eta) < abseta_tags
    pt_cond_probes = zcands.probe.pt > pt_probes
    trig_matched_tag = _trigger_match(zcands.tag, trigobjs, 30, 1)
    zcands = zcands[trig_matched_tag & pt_cond_tags & pt_cond_probes & eta_cond_tags]
    events_with_tags = dak.num(zcands.tag, axis=1) >= 1
    zcands = zcands[events_with_tags]
    trigobjs = trigobjs[events_with_tags]
    tags = zcands.tag
    probes = zcands.probe
    dr = tags.delta_r(probes)
    mass = (tags + probes).mass
    if cut_and_count:
        in_mass_window = abs(mass - 91.1876) < 30
    else:
        in_mass_window = (mass > 50) & (mass < 130)
    opposite_charge = tags.charge * probes.charge == -1
    isZ = in_mass_window & opposite_charge
    dr_condition = dr > 0.0
    all_probes = probes[isZ & dr_condition]
    pair_mass = mass[isZ & dr_condition]
    all_probes["pair_mass"] = pair_mass
    trig_matched_probe = _trigger_match(all_probes, trigobjs, trigger_pt, filterbit)
    if hlt_filter is None:
        passing_probes = all_probes[trig_matched_probe]
        failing_probes = all_probes[~trig_matched_probe]
    else:
        passing_probes = all_probes[trig_matched_probe & getattr(good_events[events_with_tags].HLT, hlt_filter)]
        failing_probes = all_probes[~(trig_matched_probe & getattr(good_events[events_with_tags].HLT, hlt_filter))]
    return passing_probes, failing_probes
