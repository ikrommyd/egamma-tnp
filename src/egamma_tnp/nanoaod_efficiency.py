from __future__ import annotations

import awkward as ak  # noqa: F401
import dask_awkward as dak
import numpy as np  # noqa: F401
from coffea.analysis_tools import Weights
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import nanoaod

from egamma_tnp._base_tagnprobe import BaseTagNProbe
from egamma_tnp.utils import calculate_photon_SC_eta, custom_delta_r
from egamma_tnp.utils.pileup import create_correction, get_pileup_weight, load_correction


class ElectronTagNProbeFromNanoAOD(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        is_photon_filter=None,
        trigger_pt=None,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        filterbit=None,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        require_event_to_pass_hlt_filter=True,
    ):
        """Electron Tag and Probe efficiency from NanoAOD and EGamma NanoAOD.

        Parameters
        ----------
        fileset: dict
            The fileset to calculate the trigger efficiencies for.
        filters: list of str or None
            The names of the filters to calculate the efficiencies for.
        is_photon_filter: list of bools, optional
            Whether the filters to calculate the efficiencies are photon filters. The default is all False.
        trigger_pt: int or float, optional
            The Pt threshold of the trigger to calculate the efficiencies over that threshold. Required for trigger efficiencies.
            The default is None.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 35.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
        tags_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the tag electrons. The default is 2.5.
        probes_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the probe electrons. The default is 2.5.
        probes_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the probe electrons. The default is 2.5.
        filterbit: int, optional
            The filterbit used to match probes with trigger objects. Required for trigger efficiencies.
            The default is None.
        cutbased_id: str, optional
            ID expression to apply to the probes. An example is "cutBased >= 2".
            If None, no cutbased ID is applied. The default is None.
        extra_zcands_mask: str, optional
            An extra mask to apply to the Z candidates. The default is None.
            Must be of the form `zcands.tag/probe.<mask> & zcands.tag/probe.<mask> & ...`.
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
        require_event_to_pass_hlt_filter : bool, optional
            Also require the event to have passed the filter HLT filter under study to consider a probe belonging to that event as passing.
            The default is True.
        """
        if filters is not None:
            if trigger_pt is None:
                from egamma_tnp.utils.misc import find_pt_threshold

                self.trigger_pt = [find_pt_threshold(filter) for filter in filters]
            else:
                self.trigger_pt = trigger_pt
                if len(self.trigger_pt) != len(filters):
                    raise ValueError("The trigger_pt list must have the same length as the filters list.")

            if is_photon_filter is None:
                self.is_photon_filter = [False for _ in filters]
            else:
                self.is_photon_filter = is_photon_filter
                if len(self.is_photon_filter) != len(filters):
                    raise ValueError("The is_photon_filter list must have the same length as the filters list.")

            if filterbit is None:
                self.filterbit = [None for _ in filters]
            else:
                self.filterbit = filterbit
                if len(self.filterbit) != len(filters):
                    raise ValueError("The filterbit list must have the same length as the filters list.")
        else:
            self.trigger_pt = None
            self.is_photon_filter = None
            self.filterbit = None

        super().__init__(
            fileset=fileset,
            filters=filters,
            tags_pt_cut=tags_pt_cut,
            probes_pt_cut=probes_pt_cut,
            tags_abseta_cut=tags_abseta_cut,
            probes_abseta_cut=probes_abseta_cut,
            cutbased_id=cutbased_id,
            extra_zcands_mask=extra_zcands_mask,
            extra_filter=extra_filter,
            extra_filter_args=extra_filter_args,
            use_sc_eta=use_sc_eta,
            use_sc_phi=use_sc_phi,
            avoid_ecal_transition_tags=avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            schemaclass=NanoAODSchema,
            default_vars=["el_pt", "el_eta", "el_phi"],
        )
        self.require_event_to_pass_hlt_filter = require_event_to_pass_hlt_filter

        if filters is not None:
            for filter, bit, pt in zip(self.filters, self.filterbit, self.trigger_pt):
                if filter.startswith("HLT_"):
                    if bit is None:
                        raise ValueError("TrigObj filerbit must be provided for all trigger filters.")
                    if pt == 0:
                        raise ValueError("A trigger Pt threshold must be provided for all trigger filters.")

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ElectronTagNProbeFromNanoAOD(Filters: {self.filters}, Number of files: {n_of_files})"

    def find_probes(self, events, cut_and_count, mass_range, vars):
        if self.use_sc_eta:
            if "superclusterEta" in events.Electron.fields:
                events["Electron", "eta_to_use"] = events.Electron.superclusterEta
            else:
                events["Electron", "superclusterEta"] = events.Electron.eta + events.Electron.deltaEtaSC
                events["Electron", "eta_to_use"] = events.Electron.superclusterEta
        else:
            events["Electron", "eta_to_use"] = events.Electron.eta
        if self.use_sc_phi:
            events["Electron", "phi_to_use"] = events.Electron.superclusterPhi
        else:
            events["Electron", "phi_to_use"] = events.Electron.phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        good_events = events[events.HLT.Ele30_WPTight_Gsf]

        ij = dak.argcartesian([good_events.Electron, good_events.Electron])
        is_not_diag = ij["0"] != ij["1"]
        i, j = dak.unzip(ij[is_not_diag])
        zcands = dak.zip({"tag": good_events.Electron[i], "probe": good_events.Electron[j]})

        pass_tight_id_tags = zcands.tag.cutBased >= 4
        if self.cutbased_id is not None:
            pass_cutbased_id_probes = eval(f"zcands.probe.{self.cutbased_id}")
        else:
            pass_cutbased_id_probes = True
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        zcands = zcands[pass_tight_id_tags & pass_cutbased_id_probes & pass_zcands_mask]

        if self.avoid_ecal_transition_tags:
            tags = zcands.tag
            pass_eta_ebeegap_tags = (abs(tags.eta_to_use) < 1.4442) | (abs(tags.eta_to_use) > 1.566)
            zcands = zcands[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            probes = zcands.probe
            pass_eta_ebeegap_probes = (abs(probes.eta_to_use) < 1.4442) | (abs(probes.eta_to_use) > 1.566)
            zcands = zcands[pass_eta_ebeegap_probes]

        passing_locs, all_probe_events = ElectronTagNProbeFromNanoAOD._process_zcands(
            zcands=zcands,
            good_events=good_events,
            trigger_pt=self.trigger_pt,
            pt_tags=self.tags_pt_cut,
            pt_probes=self.probes_pt_cut,
            abseta_tags=self.tags_abseta_cut,
            abseta_probes=self.probes_abseta_cut,
            filterbit=self.filterbit,
            cut_and_count=cut_and_count,
            mass_range=mass_range,
            filters=self.filters,
            require_event_to_pass_hlt_filter=self.require_event_to_pass_hlt_filter,
            is_photon_filter=self.is_photon_filter,
        )

        if vars == "all":
            vars_tags = [f"tag_Ele_{var}" for var in all_probe_events.tag_Ele.fields]
            vars_probes = [f"el_{var}" for var in all_probe_events.el.fields]
            extra_vars = ["PV_npvs", "Rho_fixedGridRhoAll", "Rho_fixedGridRhoFastjetAll"]
            vars = vars_tags + vars_probes + extra_vars + ["event", "run", "luminosityBlock"]
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "Pileup_nTrueInt"]

        probe_dict = {}
        for var in vars:
            if var.startswith("el_"):
                probe_dict[var] = all_probe_events["el", var.removeprefix("el_")]
            elif var.startswith("tag_Ele_"):
                probe_dict[var] = all_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
            else:
                split = var.split("_", 1)
                if len(split) == 2:
                    probe_dict[var] = all_probe_events[split[0], split[1]]
                else:
                    probe_dict[var] = all_probe_events[var]
        probe_dict.update(passing_locs)
        if not cut_and_count:
            probe_dict["pair_mass"] = all_probe_events.pair_mass

        if all_probe_events.metadata.get("isMC"):
            weights = Weights(size=None, storeIndividual=True)
            if "genWeight" in all_probe_events.fields:
                weights.add("genWeight", all_probe_events.genWeight)
            else:
                weights.add("genWeight", dak.ones_like(all_probe_events.event))
            if "LHEWeight" in all_probe_events.fields:
                weights.add("LHEWeight", all_probe_events.LHEWeight.originalXWGTUP)
            else:
                weights.add("LHEWeight", dak.ones_like(all_probe_events.event))
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.Pileup.nTrueInt, pileup_corr)
                weights.add("PUWeight", pileup_weight)
            else:
                weights.add("PUWeight", dak.ones_like(all_probe_events.event))
            probe_dict["weight"] = weights.partial_weight(include=["PUWeight", "genWeight"])
            probe_dict["weight_gen"] = weights.partial_weight(include=["genWeight"])
            probe_dict["weight_total"] = weights.weight()

        final_probe_dict = {k: v for k, v in probe_dict.items() if "to_use" not in k}
        probes = dak.zip(final_probe_dict, depth_limit=1)

        return probes

    @staticmethod
    def _trigger_match(leptons, trigobjs, pdgid, pt, filterbit):
        pass_pt = trigobjs.pt > pt
        pass_id = abs(trigobjs.id) == pdgid
        pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
        trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
        delta_r = leptons.metric_table(trigger_cands, metric=custom_delta_r)
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
        abseta_probes,
        filterbit,
        cut_and_count,
        mass_range,
        filters,
        require_event_to_pass_hlt_filter,
        is_photon_filter,
    ):
        trigobjs = good_events.TrigObj
        pt_cond_tags = zcands.tag.pt > pt_tags
        eta_cond_tags = abs(zcands.tag.eta_to_use) < abseta_tags
        pt_cond_probes = zcands.probe.pt > pt_probes
        eta_cond_probes = abs(zcands.probe.eta_to_use) < abseta_probes
        trig_matched_tag = ElectronTagNProbeFromNanoAOD._trigger_match(zcands.tag, trigobjs, 11, 30, 1)
        zcands = zcands[trig_matched_tag & pt_cond_tags & pt_cond_probes & eta_cond_tags & eta_cond_probes]
        events_with_tags = dak.num(zcands.tag, axis=1) >= 1
        zcands = zcands[events_with_tags]
        trigobjs = trigobjs[events_with_tags]
        tags = zcands.tag
        probes = zcands.probe
        dr = tags.delta_r(probes)
        mass = (tags + probes).mass
        if mass_range is None:
            if cut_and_count:
                in_mass_window = abs(mass - 91.1876) < 30
            else:
                in_mass_window = (mass > 50) & (mass < 130)
        else:
            if cut_and_count:
                in_mass_window = abs(mass - 91.1876) < mass_range
            else:
                in_mass_window = (mass > mass_range[0]) & (mass < mass_range[1])
        opposite_charge = tags.charge * probes.charge == -1
        isZ = in_mass_window & opposite_charge
        dr_condition = dr > 0.0
        zcands = zcands[isZ & dr_condition]
        good_events = good_events[events_with_tags]
        has_pair = dak.num(zcands) >= 1
        zcands = zcands[has_pair]
        trigobjs = trigobjs[has_pair]
        good_events = good_events[has_pair]
        passing_locs = {}
        if filters is not None:
            for filter, isphotonfilter, bit, pt in zip(filters, is_photon_filter, filterbit, trigger_pt):
                doclist = [x for x in good_events.TrigObj.filterBits.__doc__.split(";") if x.endswith("for Electron")]
                eledoc = doclist[0] if doclist else None
                if eledoc is not None:
                    if bit == 12 and "Leg 1" not in eledoc and "Leg 2" not in eledoc:
                        import warnings

                        warnings.warn(
                            f"You are calculating the efficiency of HLT_Ele{trigger_pt}_CaloIdVT_GsfTrkIdT in NanoAOD version < 13. Changing the filterbit to 11.",
                            stacklevel=2,
                        )
                        bit = 11
                if isphotonfilter:
                    trigobj_pdgid = 22
                else:
                    trigobj_pdgid = 11
                if filter.startswith("HLT_"):
                    is_passing_probe = ElectronTagNProbeFromNanoAOD._trigger_match(zcands.probe, trigobjs, trigobj_pdgid, pt, bit)
                else:
                    is_passing_probe = eval(f"zcands.probe.{filter}")
                if filter.startswith("HLT_") and require_event_to_pass_hlt_filter:
                    hlt_filter = filter.rsplit("_", 1)[0].split("HLT_")[1] if filter.split("HLT_")[1] not in good_events.HLT.fields else filter.split("HLT_")[1]
                    passing_locs[filter] = is_passing_probe & getattr(good_events.HLT, hlt_filter)
                else:
                    passing_locs[filter] = is_passing_probe
        all_probe_events = good_events
        all_probe_events["el"] = zcands.probe
        all_probe_events["tag_Ele"] = zcands.tag
        all_probe_events["pair_mass"] = (all_probe_events["el"] + all_probe_events["tag_Ele"]).mass

        return passing_locs, all_probe_events


class PhotonTagNProbeFromNanoAOD(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        is_electron_filter=None,
        start_from_diphotons=True,
        trigger_pt=None,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        filterbit=None,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        require_event_to_pass_hlt_filter=True,
    ):
        """Photon Tag and Probe efficiency from NanoAOD and EGamma NanoAOD.

        Parameters
        ----------
        fileset: dict
            The fileset to calculate the trigger efficiencies for.
        filters: list of str or None
            The names of the filters to calculate the efficiencies for.
        is_electron_filter: list of bools, optional
            Whether the filters to calculate the efficiencies are electron filters. The default is False.
        start_from_diphotons: bool, optional
            Whether to consider photon-photon pairs as tag-probe pairs.
            If True, it will consider photon-photon pairs as tag-probe pairs and request that the tag has an associated electron and pixel seed.
            If False, it will consider electron-photon pairs as tag-probe pairs and request that they are not associated with each other and dR > 0.1 between them.
            The default is True.
        trigger_pt: int or float, optional
            The Pt threshold of the trigger to calculate the efficiencies over that threshold.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag photons. The default is 35.
        probes_pt_cut: int or float, optional
            The Pt threshold of the probe photon to calculate efficiencies over that threshold. The default is None.
        tags_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the tag photons. The default is 2.5.
        probes_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the probe photons. The default is 2.5.
        filterbit: int, optional
            The filterbit used to match probes with trigger objects. Required for trigger efficiencies.
            The default is None.
        cutbased_id: str, optional
            ID expression to apply to the probes. An example is "cutBased >= 2".
            If None, no cutbased ID is applied. The default is None.
        extra_zcands_mask: str, optional
            An extra mask to apply to the Z candidates. The default is None.
            Must be of the form `zcands.tag/probe.<mask> & zcands.tag/probe.<mask> & ...`.
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
        require_event_to_pass_hlt_filter : bool, optional
            Also require the event to have passed the filter HLT filter under study to consider a probe belonging to that event as passing.
            The default is True.
        """
        if filters is not None:
            if trigger_pt is None:
                from egamma_tnp.utils.misc import find_pt_threshold

                self.trigger_pt = [find_pt_threshold(filter) for filter in filters]
            else:
                self.trigger_pt = trigger_pt
                if len(self.trigger_pt) != len(filters):
                    raise ValueError("The trigger_pt list must have the same length as the filters list.")

            if is_electron_filter is None:
                self.is_electron_filter = [False for _ in filters]
            else:
                self.is_electron_filter = is_electron_filter
                if len(self.is_electron_filter) != len(filters):
                    raise ValueError("The is_electron_filter list must have the same length as the filters list.")

            if filterbit is None:
                self.filterbit = [None for _ in filters]
            else:
                self.filterbit = filterbit
                if len(self.filterbit) != len(filters):
                    raise ValueError("The filterbit list must have the same length as the filters list.")
        else:
            self.trigger_pt = None
            self.is_electron_filter = None
            self.filterbit = None

        super().__init__(
            fileset=fileset,
            filters=filters,
            tags_pt_cut=tags_pt_cut,
            probes_pt_cut=probes_pt_cut,
            tags_abseta_cut=tags_abseta_cut,
            probes_abseta_cut=probes_abseta_cut,
            cutbased_id=cutbased_id,
            extra_zcands_mask=extra_zcands_mask,
            extra_filter=extra_filter,
            extra_filter_args=extra_filter_args,
            use_sc_eta=use_sc_eta,
            use_sc_phi=use_sc_phi,
            avoid_ecal_transition_tags=avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=avoid_ecal_transition_probes,
            schemaclass=NanoAODSchema,
            default_vars=["ph_pt", "ph_eta", "ph_phi"],
        )
        self.start_from_diphotons = start_from_diphotons
        self.require_event_to_pass_hlt_filter = require_event_to_pass_hlt_filter

        if filters is not None:
            for filter, bit, pt in zip(self.filters, self.filterbit, self.trigger_pt):
                if filter.startswith("HLT_"):
                    if bit is None:
                        raise ValueError("TrigObj filerbit must be provided for all trigger filters.")
                    if pt == 0:
                        raise ValueError("A trigger Pt threshold must be provided for all trigger filters.")

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"PhotonTagNProbeFromNanoAOD(Filters: {self.filters}, Number of files: {n_of_files})"

    def find_probes(self, events, cut_and_count, mass_range, vars):
        # TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
        photon_dict = {field: events.Photon[field] for field in events.Photon.fields} | {
            "mass": dak.zeros_like(events.Photon.pt),
            "charge": dak.zeros_like(events.Photon.pt),
        }
        events["Photon"] = dak.zip(photon_dict, with_name="Photon", behavior=nanoaod.behavior)

        if self.use_sc_eta:
            if "superclusterEta" not in events.Photon.fields:
                events["Photon", "superclusterEta"] = calculate_photon_SC_eta(events.Photon, events.PV)
            events["Photon", "eta_to_use"] = events.Photon.superclusterEta
            if "superclusterEta" in events.Electron.fields:
                events["Electron", "eta_to_use"] = events.Electron.superclusterEta
            else:
                events["Electron", "eta_to_use"] = events.Electron.eta + events.Electron.deltaEtaSC
        else:
            events["Photon", "eta_to_use"] = events.Photon.eta
            events["Electron", "eta_to_use"] = events.Electron.eta
        if self.use_sc_phi:
            events["Photon", "phi_to_use"] = events.Photon.superclusterPhi
            events["Electron", "phi_to_use"] = events.Electron.superclusterPhi
        else:
            events["Photon", "phi_to_use"] = events.Photon.phi
            events["Electron", "phi_to_use"] = events.Electron.phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        # keep until new coffea release
        events["Photon", "charge"] = 0.0 * events.Photon.pt

        good_events = events[events.HLT.Ele30_WPTight_Gsf]

        if self.start_from_diphotons:
            ij = dak.argcartesian([good_events.Photon, good_events.Photon])
            is_not_diag = ij["0"] != ij["1"]
            i, j = dak.unzip(ij[is_not_diag])
            zcands = dak.zip({"tag": good_events.Photon[i], "probe": good_events.Photon[j]})
            pass_tight_id_tags = zcands.tag.cutBased >= 3
        else:
            ij = dak.argcartesian({"tag": good_events.Electron, "probe": good_events.Photon})
            tnp = dak.cartesian({"tag": good_events.Electron, "probe": good_events.Photon})
            probe_is_not_tag = (tnp.probe.electronIdx != ij.tag) & (tnp.tag.delta_r(tnp.probe) > 0.1)
            zcands = tnp[probe_is_not_tag]
            pass_tight_id_tags = zcands.tag.cutBased >= 4

        if self.cutbased_id is not None:
            pass_cutbased_id_probes = eval(f"zcands.probe.{self.cutbased_id}")
        else:
            pass_cutbased_id_probes = True
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        zcands = zcands[pass_tight_id_tags & pass_cutbased_id_probes & pass_zcands_mask]

        if self.avoid_ecal_transition_tags:
            tags = zcands.tag
            pass_eta_ebeegap_tags = (abs(tags.eta_to_use) < 1.4442) | (abs(tags.eta_to_use) > 1.566)
            zcands = zcands[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            probes = zcands.probe
            pass_eta_ebeegap_probes = (abs(probes.eta_to_use) < 1.4442) | (abs(probes.eta_to_use) > 1.566)
            zcands = zcands[pass_eta_ebeegap_probes]

        passing_locs, all_probe_events = PhotonTagNProbeFromNanoAOD._process_zcands(
            zcands=zcands,
            good_events=good_events,
            trigger_pt=self.trigger_pt,
            pt_tags=self.tags_pt_cut,
            pt_probes=self.probes_pt_cut,
            abseta_tags=self.tags_abseta_cut,
            abseta_probes=self.probes_abseta_cut,
            filterbit=self.filterbit,
            cut_and_count=cut_and_count,
            mass_range=mass_range,
            filters=self.filters,
            require_event_to_pass_hlt_filter=self.require_event_to_pass_hlt_filter,
            is_electron_filter=self.is_electron_filter,
            start_from_diphotons=self.start_from_diphotons,
        )

        if vars == "all":
            vars_tags = [f"tag_Ele_{var}" for var in all_probe_events.tag_Ele.fields]
            vars_probes = [f"ph_{var}" for var in all_probe_events.ph.fields]
            extra_vars = ["PV_npvs", "Rho_fixedGridRhoAll", "Rho_fixedGridRhoFastjetAll"]
            vars = vars_tags + vars_probes + extra_vars + ["event", "run", "luminosityBlock"]
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "Pileup_nTrueInt"]

        probe_dict = {}
        for var in vars:
            if var.startswith("ph_"):
                probe_dict[var] = all_probe_events["ph", var.removeprefix("ph_")]
            elif var.startswith("tag_Ele_"):
                probe_dict[var] = all_probe_events["tag_Ele", var.removeprefix("tag_Ele_")]
            else:
                split = var.split("_", 1)
                if len(split) == 2:
                    probe_dict[var] = all_probe_events[split[0], split[1]]
                else:
                    probe_dict[var] = all_probe_events[var]
        probe_dict.update(passing_locs)
        if not cut_and_count:
            probe_dict["pair_mass"] = all_probe_events.pair_mass

        if all_probe_events.metadata.get("isMC"):
            weights = Weights(size=None, storeIndividual=True)
            if "genWeight" in all_probe_events.fields:
                weights.add("genWeight", all_probe_events.genWeight)
            else:
                weights.add("genWeight", dak.ones_like(all_probe_events.event))
            if "LHEWeight" in all_probe_events.fields:
                weights.add("LHEWeight", all_probe_events.LHEWeight.originalXWGTUP)
            else:
                weights.add("LHEWeight", dak.ones_like(all_probe_events.event))
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.Pileup.nTrueInt, pileup_corr)
                weights.add("PUWeight", pileup_weight)
            else:
                weights.add("PUWeight", dak.ones_like(all_probe_events.event))
            probe_dict["weight"] = weights.partial_weight(include=["PUWeight", "genWeight"])
            probe_dict["weight_gen"] = weights.partial_weight(include=["genWeight"])
            probe_dict["weight_total"] = weights.weight()

        final_probe_dict = {k: v for k, v in probe_dict.items() if "to_use" not in k}
        probes = dak.zip(final_probe_dict, depth_limit=1)

        final_probe_dict = {k: v for k, v in probe_dict.items() if "to_use" not in k}
        probes = dak.zip(final_probe_dict, depth_limit=1)

        return probes

    @staticmethod
    def _trigger_match(leptons, trigobjs, pdgid, pt, filterbit):
        pass_pt = trigobjs.pt > pt
        pass_id = abs(trigobjs.id) == pdgid
        pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
        trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
        delta_r = leptons.metric_table(trigger_cands, metric=custom_delta_r)
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
        abseta_probes,
        filterbit,
        cut_and_count,
        mass_range,
        filters,
        require_event_to_pass_hlt_filter,
        is_electron_filter,
        start_from_diphotons,
    ):
        trigobjs = good_events.TrigObj
        pt_cond_tags = zcands.tag.pt > pt_tags
        eta_cond_tags = abs(zcands.tag.eta_to_use) < abseta_tags
        pt_cond_probes = zcands.probe.pt > pt_probes
        eta_cond_probes = abs(zcands.probe.eta_to_use) < abseta_probes
        if start_from_diphotons:
            has_matched_electron_tags = (zcands.tag.electronIdx != -1) & (zcands.tag.pixelSeed)
            trig_matched_tag = PhotonTagNProbeFromNanoAOD._trigger_match(zcands.tag.matched_electron, trigobjs, 11, 30, 1)
        else:
            has_matched_electron_tags = True
            trig_matched_tag = PhotonTagNProbeFromNanoAOD._trigger_match(zcands.tag, trigobjs, 11, 30, 1)
        zcands = zcands[has_matched_electron_tags & trig_matched_tag & pt_cond_tags & pt_cond_probes & eta_cond_tags & eta_cond_probes]
        events_with_tags = dak.num(zcands.tag, axis=1) >= 1
        zcands = zcands[events_with_tags]
        trigobjs = trigobjs[events_with_tags]
        tags = zcands.tag
        probes = zcands.probe
        dr = tags.delta_r(probes)
        mass = (tags + probes).mass
        if mass_range is None:
            if cut_and_count:
                in_mass_window = abs(mass - 91.1876) < 30
            else:
                in_mass_window = (mass > 50) & (mass < 130)
        else:
            if cut_and_count:
                in_mass_window = abs(mass - 91.1876) < mass_range
            else:
                in_mass_window = (mass > mass_range[0]) & (mass < mass_range[1])
        # Use this unless we choose to pixel match the probes as well
        opposite_charge = True
        # opposite_charge = tags.charge * probes.charge == -1
        isZ = in_mass_window & opposite_charge
        dr_condition = dr > 0.0
        zcands = zcands[isZ & dr_condition]
        good_events = good_events[events_with_tags]
        has_pair = dak.num(zcands) >= 1
        zcands = zcands[has_pair]
        trigobjs = trigobjs[has_pair]
        good_events = good_events[has_pair]
        passing_locs = {}
        if filters is not None:
            for filter, iselectronfilter, bit, pt in zip(filters, is_electron_filter, filterbit, trigger_pt):
                if iselectronfilter:
                    trigobj_pdgid = 11
                else:
                    trigobj_pdgid = 22
                if filter.startswith("HLT_"):
                    is_passing_probe = PhotonTagNProbeFromNanoAOD._trigger_match(zcands.probe, trigobjs, trigobj_pdgid, pt, bit)
                else:
                    is_passing_probe = eval(f"zcands.probe.{filter}")
                if filter.startswith("HLT_") and require_event_to_pass_hlt_filter:
                    hlt_filter = filter.rsplit("_", 1)[0].split("HLT_")[1] if filter.split("HLT_")[1] not in good_events.HLT.fields else filter.split("HLT_")[1]
                    passing_locs[filter] = is_passing_probe & getattr(good_events.HLT, hlt_filter)
                else:
                    passing_locs[filter] = is_passing_probe
        all_probe_events = good_events
        all_probe_events["ph"] = zcands.probe
        all_probe_events["tag_Ele"] = zcands.tag
        all_probe_events["pair_mass"] = (all_probe_events["ph"] + all_probe_events["tag_Ele"]).mass

        return passing_locs, all_probe_events
