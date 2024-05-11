import dask_awkward as dak
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoAODSchema

from egamma_tnp._base_tagnprobe import BaseTagNProbe
from egamma_tnp.utils import custom_delta_r


class ElectronTagNProbeFromNanoAOD(BaseTagNProbe):
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
        """Electron Tag and Probe efficiency from NanoAOD and EGamma NanoAOD.
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
            default_vars=["el_pt", "el_eta", "el_phi"],
        )
        self.for_trigger = for_trigger
        self.egm_nano = egm_nano
        self.filterbit = filterbit
        self.hlt_filter = hlt_filter

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ElectronTagNProbeFromNanoAOD({self.filter}, Number of files: {n_of_files}, Golden JSON: {self.goldenjson})"

    def _find_probes(self, events, cut_and_count, vars):
        if self.use_sc_eta:
            if self.egm_nano:
                events["Electron", "eta_to_use"] = events.Electron.superclusterEta
            else:
                events["Electron", "eta_to_use"] = events.Electron.eta + events.Electron.deltaEtaSC
        else:
            events["Electron", "eta_to_use"] = events.Electron.eta
        if self.use_sc_phi:
            events["Electron", "phi_to_use"] = events.Electron.superclusterPhi
        else:
            events["Electron", "phi_to_use"] = events.Electron.phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        good_events, good_locations = ElectronTagNProbeFromNanoAOD._filter_events(events, self.cutbased_id)
        ele_for_tnp = good_events.Electron[good_locations]
        zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])

        if self.avoid_ecal_transition_tags:
            tags1 = zcands1.tag
            pass_eta_ebeegap_tags1 = (abs(tags1.eta_to_use) < 1.4442) | (abs(tags1.eta_to_use) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_tags1]
        if self.avoid_ecal_transition_probes:
            probes1 = zcands1.probe
            pass_eta_ebeegap_probes1 = (abs(probes1.eta_to_use) < 1.4442) | (abs(probes1.eta_to_use) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_probes1]

        p1, f1 = ElectronTagNProbeFromNanoAOD._process_zcands(
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

        p1["el"] = p1.Electron[:, 1]
        f1["el"] = f1.Electron[:, 1]
        p1["tag_Ele"] = p1.Electron[:, 0]
        f1["tag_Ele"] = f1.Electron[:, 0]
        p1["pair_mass"] = (p1["el"] + p1["tag_Ele"]).mass
        f1["pair_mass"] = (f1["el"] + f1["tag_Ele"]).mass

        zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])

        if self.avoid_ecal_transition_tags:
            tags2 = zcands2.tag
            pass_eta_ebeegap_tags2 = (abs(tags2.eta_to_use) < 1.4442) | (abs(tags2.eta_to_use) > 1.566)
            zcands2 = zcands2[pass_eta_ebeegap_tags2]
        if self.avoid_ecal_transition_probes:
            probes2 = zcands2.probe
            pass_eta_ebeegap_probes2 = (abs(probes2.eta_to_use) < 1.4442) | (abs(probes2.eta_to_use) > 1.566)
            zcands2 = zcands2[pass_eta_ebeegap_probes2]

        p2, f2 = ElectronTagNProbeFromNanoAOD._process_zcands(
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

        p2["el"] = p2.Electron[:, 0]
        f2["el"] = f2.Electron[:, 0]
        p2["tag_Ele"] = p2.Electron[:, 1]
        f2["tag_Ele"] = f2.Electron[:, 1]
        p2["pair_mass"] = (p2["el"] + p2["tag_Ele"]).mass
        f2["pair_mass"] = (f2["el"] + f2["tag_Ele"]).mass

        passing_probe_events, failing_probe_events = dak.concatenate([p1, p2]), dak.concatenate([f1, f2])

        passing_probe_dict = {}
        failing_probe_dict = {}
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
        passing_probes = dak.zip(passing_probe_dict, depth_limit=1)
        failing_probes = dak.zip(failing_probe_dict, depth_limit=1)

        return passing_probes, failing_probes

    @staticmethod
    def _filter_events(events, cutbased_id):
        pass_hlt = events.HLT.Ele30_WPTight_Gsf
        two_electrons = dak.num(events.Electron) == 2
        abs_eta = abs(events.Electron.eta_to_use)
        if cutbased_id:
            pass_tight_id = events.Electron.cutBased == cutbased_id
        else:
            pass_tight_id = True
        pass_eta = abs_eta <= 2.5
        pass_selection = pass_hlt & two_electrons & pass_eta & pass_tight_id
        n_of_tags = dak.sum(pass_selection, axis=1)
        good_events = events[n_of_tags == 2]
        good_locations = pass_selection[n_of_tags == 2]
        return good_events, good_locations

    @staticmethod
    def _trigger_match(electrons, trigobjs, pt, filterbit):
        pass_pt = trigobjs.pt > pt
        pass_id = abs(trigobjs.id) == 11
        pass_filterbit = trigobjs.filterBits & (0x1 << filterbit) > 0
        trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
        delta_r = electrons.metric_table(trigger_cands, metric=custom_delta_r)
        pass_delta_r = delta_r < 0.1
        n_of_trigger_matches = dak.sum(pass_delta_r, axis=2)
        trig_matched_locs = n_of_trigger_matches >= 1
        return trig_matched_locs

    @staticmethod
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
        eta_cond_tags = abs(zcands.tag.eta_to_use) < abseta_tags
        pt_cond_probes = zcands.probe.pt > pt_probes
        trig_matched_tag = ElectronTagNProbeFromNanoAOD._trigger_match(zcands.tag, trigobjs, 30, 1)
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
        trig_matched_probe = ElectronTagNProbeFromNanoAOD._trigger_match(all_probes, trigobjs, trigger_pt, filterbit)
        good_events = good_events[events_with_tags]
        if hlt_filter is None:
            has_passing_probe = dak.fill_none(dak.firsts(trig_matched_probe), False)
            has_failing_probe = dak.fill_none(dak.firsts(~trig_matched_probe), False)
        else:
            has_passing_probe = dak.fill_none(dak.firsts(trig_matched_probe & getattr(good_events.HLT, hlt_filter)), False)
            has_failing_probe = dak.fill_none(
                dak.firsts(~(trig_matched_probe & getattr(good_events.HLT, hlt_filter))),
                False,
            )
        passing_probe_events = good_events[has_passing_probe]
        failing_probe_events = good_events[has_failing_probe]
        return passing_probe_events, failing_probe_events


class PhotonTagNProbeFromNanoAOD(BaseTagNProbe):
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
        """Photon Tag and Probe efficiency from NanoAOD and EGamma NanoAOD.
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
            default_vars=["ph_pt", "ph_eta", "ph_phi"],
        )
        self.for_trigger = for_trigger
        self.egm_nano = egm_nano
        self.filterbit = filterbit
        self.hlt_filter = hlt_filter

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"PhotonTagNProbeFromNanoAOD({self.filter}, Number of files: {n_of_files}, Golden JSON: {self.goldenjson})"

    def _find_probes(self, events, cut_and_count, vars):
        if self.use_sc_eta:
            if self.egm_nano:
                events["Electron", "eta_to_use"] = events.Electron.superclusterEta
            else:
                events["Electron", "eta_to_use"] = events.Electron.eta + events.Electron.deltaEtaSC
        else:
            events["Electron", "eta_to_use"] = events.Electron.eta
        if self.use_sc_phi:
            events["Electron", "phi_to_use"] = events.Electron.superclusterPhi
        else:
            events["Electron", "phi_to_use"] = events.Electron.phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        good_events, good_locations = PhotonTagNProbeFromNanoAOD._filter_events(events, self.cutbased_id)
        ele_for_tnp = good_events.Electron[good_locations]
        zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])

        if self.avoid_ecal_transition_tags:
            tags1 = zcands1.tag
            pass_eta_ebeegap_tags1 = (abs(tags1.eta_to_use) < 1.4442) | (abs(tags1.eta_to_use) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_tags1]
        if self.avoid_ecal_transition_probes:
            probes1 = zcands1.probe
            pass_eta_ebeegap_probes1 = (abs(probes1.eta_to_use) < 1.4442) | (abs(probes1.eta_to_use) > 1.566)
            zcands1 = zcands1[pass_eta_ebeegap_probes1]

        p1, f1 = PhotonTagNProbeFromNanoAOD._process_zcands(
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

        p1["el"] = p1.Electron[:, 1]
        f1["el"] = f1.Electron[:, 1]
        p1["tag_Ele"] = p1.Electron[:, 0]
        f1["tag_Ele"] = f1.Electron[:, 0]
        p1["pair_mass"] = (p1["el"] + p1["tag_Ele"]).mass
        f1["pair_mass"] = (f1["el"] + f1["tag_Ele"]).mass

        zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])

        if self.avoid_ecal_transition_tags:
            tags2 = zcands2.tag
            pass_eta_ebeegap_tags2 = (abs(tags2.eta_to_use) < 1.4442) | (abs(tags2.eta_to_use) > 1.566)
            zcands2 = zcands2[pass_eta_ebeegap_tags2]
        if self.avoid_ecal_transition_probes:
            probes2 = zcands2.probe
            pass_eta_ebeegap_probes2 = (abs(probes2.eta_to_use) < 1.4442) | (abs(probes2.eta_to_use) > 1.566)
            zcands2 = zcands2[pass_eta_ebeegap_probes2]

        p2, f2 = PhotonTagNProbeFromNanoAOD._process_zcands(
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

        p2["el"] = p2.Electron[:, 0]
        f2["el"] = f2.Electron[:, 0]
        p2["tag_Ele"] = p2.Electron[:, 1]
        f2["tag_Ele"] = f2.Electron[:, 1]
        p2["pair_mass"] = (p2["el"] + p2["tag_Ele"]).mass
        f2["pair_mass"] = (f2["el"] + f2["tag_Ele"]).mass

        passing_probe_events, failing_probe_events = dak.concatenate([p1, p2]), dak.concatenate([f1, f2])

        passing_probe_dict = {}
        failing_probe_dict = {}
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
        passing_probes = dak.zip(passing_probe_dict, depth_limit=1)
        failing_probes = dak.zip(failing_probe_dict, depth_limit=1)

        return passing_probes, failing_probes

    @staticmethod
    def _filter_events(events, cutbased_id):
        pass_hlt = events.HLT.Ele30_WPTight_Gsf
        two_electrons = dak.num(events.Electron) == 2
        abs_eta = abs(events.Electron.eta_to_use)
        if cutbased_id:
            pass_tight_id = events.Electron.cutBased == cutbased_id
        else:
            pass_tight_id = True
        pass_eta = abs_eta <= 2.5
        pass_selection = pass_hlt & two_electrons & pass_eta & pass_tight_id
        n_of_tags = dak.sum(pass_selection, axis=1)
        good_events = events[n_of_tags == 2]
        good_locations = pass_selection[n_of_tags == 2]
        return good_events, good_locations

    @staticmethod
    def _trigger_match(electrons, trigobjs, pt, filterbit):
        pass_pt = trigobjs.pt > pt
        pass_id = abs(trigobjs.id) == 11
        pass_filterbit = trigobjs.filterBits & (0x1 << filterbit) > 0
        trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
        delta_r = electrons.metric_table(trigger_cands, metric=custom_delta_r)
        pass_delta_r = delta_r < 0.1
        n_of_trigger_matches = dak.sum(pass_delta_r, axis=2)
        trig_matched_locs = n_of_trigger_matches >= 1
        return trig_matched_locs

    @staticmethod
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
        eta_cond_tags = abs(zcands.tag.eta_to_use) < abseta_tags
        pt_cond_probes = zcands.probe.pt > pt_probes
        trig_matched_tag = PhotonTagNProbeFromNanoAOD._trigger_match(zcands.tag, trigobjs, 30, 1)
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
        trig_matched_probe = PhotonTagNProbeFromNanoAOD._trigger_match(all_probes, trigobjs, trigger_pt, filterbit)
        good_events = good_events[events_with_tags]
        if hlt_filter is None:
            has_passing_probe = dak.fill_none(dak.firsts(trig_matched_probe), False)
            has_failing_probe = dak.fill_none(dak.firsts(~trig_matched_probe), False)
        else:
            has_passing_probe = dak.fill_none(dak.firsts(trig_matched_probe & getattr(good_events.HLT, hlt_filter)), False)
            has_failing_probe = dak.fill_none(
                dak.firsts(~(trig_matched_probe & getattr(good_events.HLT, hlt_filter))),
                False,
            )
        passing_probe_events = good_events[has_passing_probe]
        failing_probe_events = good_events[has_failing_probe]
        return passing_probe_events, failing_probe_events
