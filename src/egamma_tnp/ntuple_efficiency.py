from __future__ import annotations

import awkward as ak  # noqa: F401
import dask_awkward as dak
import numpy as np  # noqa: F401
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import BaseSchema

from egamma_tnp._base_tagnprobe import BaseTagNProbe
from egamma_tnp.utils.pileup import create_correction, get_pileup_weight, load_correction


class ElectronTagNProbeFromMiniNTuples(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
    ):
        """Electron Tag and Probe efficiency from E/Gamma NTuples from MiniAOD.

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filters: list of str or None
                The name of the filters to calculate the efficiencies for.
            tags_pt_cut: int or float, optional
                The Pt cut to apply to the tag electrons. The default is 35.
            probes_pt_cut: int or float, optional
                The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
            tags_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the tag electrons. The default is 2.5.
            probes_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the probe electrons. The default is 2.5.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
                If None, no cutbased ID is applied. The default is None.
            extra_zcands_mask: str, optional
                An extra mask to apply to the Z candidates. The default is None.
                Must be of the form `events.<mask> & events.<mask> & ...`.
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
        """
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
            schemaclass=BaseSchema,
            default_vars=["el_pt", "el_eta", "el_phi"],
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ElectronTagNProbeFromMiniNTuples(Filters: {self.filters}, Number of files: {n_of_files})"

    def _find_passing_events(self, events, cut_and_count, mass_range):
        pass_pt_probes = events.el_pt > self.probes_pt_cut
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if mass_range is not None:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < mass_range
            else:
                in_mass_window = (events.pair_mass > mass_range[0]) & (events.pair_mass < mass_range[1])
        else:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < 30
            else:
                in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
        all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
        if self.filters is not None:
            passing_locs = {filter: (all_probe_events[filter] == 1) for filter in self.filters}
        else:
            passing_locs = {}

        return passing_locs, all_probe_events

    def find_probes(self, events, cut_and_count, mass_range, vars):
        if self.use_sc_eta:
            events["el_eta_to_use"] = events.el_sc_eta
            events["tag_Ele_eta_to_use"] = events.tag_sc_eta
        else:
            events["el_eta_to_use"] = events.el_eta
            events["tag_Ele_eta_to_use"] = events.tag_Ele_eta
        if self.use_sc_phi:
            events["el_phi_to_use"] = events.el_sc_phi
        else:
            events["el_phi_to_use"] = events.el_phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.el_eta_to_use) < 1.4442) | (abs(events.el_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < self.tags_abseta_cut
        pass_abseta_probes = abs(events.el_eta_to_use) < self.probes_abseta_cut
        opposite_charge = events.tag_Ele_q * events.el_q == -1
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & opposite_charge & pass_zcands_mask]

        passing_locs, all_probe_events = self._find_passing_events(events, cut_and_count=cut_and_count, mass_range=mass_range)

        if vars == "all":
            vars_tags = [v for v in all_probe_events.fields if v.startswith("tag_Ele_")]
            vars_probes = [v for v in all_probe_events.fields if v.startswith("el_")]
            vars = vars_tags + vars_probes + ["event", "run", "lumi"] + [x for x in all_probe_events.fields if "weight" in x or "Weight" in x]
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "truePU"]

        if cut_and_count:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs)
        else:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs | {"pair_mass": all_probe_events["pair_mass"]})

        if all_probe_events.metadata.get("isMC"):
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.truePU, pileup_corr)
                probes["weight"] = pileup_weight

        return probes


class PhotonTagNProbeFromMiniNTuples(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
    ):
        """Photon Tag and Probe efficiency from E/Gamma NTuples from MiniAOD

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filters: list of str or None
                The name of the filters to calculate the efficiencies for.
            tags_pt_cut: int or float, optional
                The Pt cut to apply to the tag photons. The default is 35.
            probes_pt_cut: int or float, optional
                The Pt threshold of the probe photon to calculate efficiencies over that threshold. The default is None.
            tags_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the tag photons. The default is 2.5.
            probes_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the probe photons. The default is 2.5.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
                If None, no cutbased ID is applied. The default is None.
            extra_zcands_mask: str, optional
                An extra mask to apply to the Z candidates. The default is None.
                Must be of the form `events.<mask> & events.<mask> & ...`.
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
        """
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
            schemaclass=BaseSchema,
            default_vars=["ph_et", "ph_eta", "ph_phi"],
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"PhotonTagNProbeFromMiniNTuples(Filters: {self.filters}, Number of files: {n_of_files})"

    def _find_passing_events(self, events, cut_and_count, mass_range):
        pass_pt_probes = events.ph_et > self.probes_pt_cut
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if mass_range is not None:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < mass_range
            else:
                in_mass_window = (events.pair_mass > mass_range[0]) & (events.pair_mass < mass_range[1])
        else:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < 30
            else:
                in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
        all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
        if self.filters is not None:
            passing_locs = {filter: (all_probe_events[filter] == 1) for filter in self.filters}
        else:
            passing_locs = {}

        return passing_locs, all_probe_events

    def find_probes(self, events, cut_and_count, mass_range, vars):
        if self.use_sc_eta:
            events["ph_eta_to_use"] = events.ph_sc_eta
            events["tag_Ele_eta_to_use"] = events.tag_sc_eta
        else:
            events["ph_eta_to_use"] = events.ph_eta
            events["tag_Ele_eta_to_use"] = events.tag_Ele_eta
        if self.use_sc_phi:
            events["ph_phi_to_use"] = events.ph_sc_phi
        else:
            events["ph_phi_to_use"] = events.ph_phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.ph_eta_to_use) < 1.4442) | (abs(events.ph_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < self.tags_abseta_cut
        pass_abseta_probes = abs(events.ph_eta_to_use) < self.probes_abseta_cut
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & pass_zcands_mask]

        passing_locs, all_probe_events = self._find_passing_events(events, cut_and_count=cut_and_count, mass_range=mass_range)

        if vars == "all":
            vars_tags = [v for v in all_probe_events.fields if v.startswith("tag_Ele_")]
            vars_probes = [v for v in all_probe_events.fields if v.startswith("el_")]
            vars = vars_tags + vars_probes + ["event", "run", "lumi"] + [x for x in all_probe_events.fields if "weight" in x or "Weight" in x]
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "truePU"]

        if cut_and_count:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs)
        else:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs | {"pair_mass": all_probe_events["pair_mass"]})

        if all_probe_events.metadata.get("isMC"):
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.truePU, pileup_corr)
                probes["weight"] = pileup_weight

        return probes


class ElectronTagNProbeFromNanoNTuples(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
    ):
        """Electron Tag and Probe efficiency from E/Gamma NTuples from NanoAOD.

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filters: list of str or None
                The name of the filters to calculate the efficiencies for.
            tags_pt_cut: int or float, optional
                The Pt cut to apply to the tag electrons. The default is 35.
            probes_pt_cut: int or float, optional
                The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
            tags_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the tag electrons. The default is 2.5.
            probes_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the probe electrons. The default is 2.5.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
                If None, no cutbased ID is applied. The default is None.
            extra_zcands_mask: str, optional
                An extra mask to apply to the Z candidates. The default is None.
                Must be of the form `events.<mask> & events.<mask> & ...`.
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
        """
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
            schemaclass=BaseSchema,
            default_vars=["el_pt", "el_eta", "el_phi"],
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ElectronTagNProbeFromNanoNTuples(Filters: {self.filters}, Number of files: {n_of_files})"

    def _find_passing_events(self, events, cut_and_count, mass_range):
        pass_pt_probes = events.el_pt > self.probes_pt_cut
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if mass_range is not None:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < mass_range
            else:
                in_mass_window = (events.pair_mass > mass_range[0]) & (events.pair_mass < mass_range[1])
        else:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < 30
            else:
                in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
        all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
        if self.filters is not None:
            passing_locs = {filter: (all_probe_events[filter] == 1) for filter in self.filters}
        else:
            passing_locs = {}

        return passing_locs, all_probe_events

    def find_probes(self, events, cut_and_count, mass_range, vars):
        if self.use_sc_eta:
            if "el_superclusterEta" in events.fields:
                events["el_eta_to_use"] = events.el_superclusterEta
            else:
                events["el_eta_to_use"] = events.el_eta + events.el_deltaEtaSC
            if "tag_Ele_superclusterEta" in events.fields:
                events["tag_Ele_eta_to_use"] = events.tag_Ele_superclusterEta
            else:
                events["tag_Ele_eta_to_use"] = events.tag_Ele_eta + events.tag_Ele_deltaEtaSC
        else:
            events["el_eta_to_use"] = events.el_eta
            events["tag_Ele_eta_to_use"] = events.tag_Ele_eta
        if self.use_sc_phi:
            events["el_phi_to_use"] = events.el_superclusterPhi
        else:
            events["el_phi_to_use"] = events.el_phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.el_eta_to_use) < 1.4442) | (abs(events.el_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < self.tags_abseta_cut
        pass_abseta_probes = abs(events.el_eta_to_use) < self.probes_abseta_cut
        opposite_charge = events.tag_Ele_charge * events.el_charge == -1
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & opposite_charge & pass_zcands_mask]

        passing_locs, all_probe_events = self._find_passing_events(events, cut_and_count=cut_and_count, mass_range=mass_range)

        if vars == "all":
            vars_tags = [v for v in all_probe_events.fields if v.startswith("tag_Ele_")]
            vars_probes = [v for v in all_probe_events.fields if v.startswith("el_")]
            vars = (
                vars_tags
                + vars_probes
                + ["event", "run", "luminosityBlock"]
                + [
                    x
                    for x in all_probe_events.fields
                    if "weight" in x or "Weight" in x or x == "PV_npvs" or x == "Rho_fixedGridRhoAll" or x == "Rho_fixedGridRhoFastjetAll"
                ]
            )
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "Pileup_nTrueInt"]

        if cut_and_count:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs)
        else:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs | {"pair_mass": all_probe_events["pair_mass"]})

        if all_probe_events.metadata.get("isMC") and "weight" not in vars:
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.Pileup_nTrueInt, pileup_corr)
                probes["weight"] = pileup_weight

        return probes


class PhotonTagNProbeFromNanoNTuples(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filters,
        *,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        cutbased_id=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
    ):
        """Photon Tag and Probe efficiency from E/Gamma NTuples from NanoAOD.

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filters: list of str or None
                The name of the filters to calculate the efficiencies for.
            tags_pt_cut: int or float, optional
                The Pt cut to apply to the tag photons. The default is 35.
            probes_pt_cut: int or float, optional
                The Pt threshold of the probe photon to calculate efficiencies over that threshold. The default is None.
            tags_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the tag photons. The default is 2.5.
            probes_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the probe photons. The default is 2.5.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
                If None, no cutbased ID is applied. The default is None.
            extra_zcands_mask: str, optional
                An extra mask to apply to the Z candidates. The default is None.
                Must be of the form `events.<mask> & events.<mask> & ...`.
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
        """
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
            schemaclass=BaseSchema,
            default_vars=["ph_pt", "ph_eta", "ph_phi"],
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"PhotonTagNProbeFromNanoNTuples(Filters: {self.filters}, Number of files: {n_of_files})"

    def _find_passing_events(self, events, cut_and_count, mass_range):
        pass_pt_probes = events.ph_pt > self.probes_pt_cut
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if mass_range is not None:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < mass_range
            else:
                in_mass_window = (events.pair_mass > mass_range[0]) & (events.pair_mass < mass_range[1])
        else:
            if cut_and_count:
                in_mass_window = abs(events.pair_mass - 91.1876) < 30
            else:
                in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
        all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
        if self.filters is not None:
            passing_locs = {filter: (all_probe_events[filter] == 1) for filter in self.filters}
        else:
            passing_locs = {}

        return passing_locs, all_probe_events

    def find_probes(self, events, cut_and_count, mass_range, vars):
        if self.use_sc_eta:
            events["ph_eta_to_use"] = events.ph_superclusterEta
            events["tag_Ele_eta_to_use"] = events.tag_Ele_superclusterEta
        else:
            events["ph_eta_to_use"] = events.ph_eta
            events["tag_Ele_eta_to_use"] = events.tag_Ele_eta
        if self.use_sc_phi:
            events["ph_phi_to_use"] = events.ph_superclusterPhi
        else:
            events["ph_phi_to_use"] = events.ph_phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta_to_use) < 1.4442) | (abs(events.tag_Ele_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.ph_eta_to_use) < 1.4442) | (abs(events.ph_eta_to_use) > 1.566)
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta_to_use) < self.tags_abseta_cut
        pass_abseta_probes = abs(events.ph_eta_to_use) < self.probes_abseta_cut
        if self.extra_zcands_mask is not None:
            pass_zcands_mask = eval(self.extra_zcands_mask)
        else:
            pass_zcands_mask = True
        events = events[pass_pt_tags & pass_abseta_tags & pass_abseta_probes & pass_zcands_mask]

        passing_locs, all_probe_events = self._find_passing_events(events, cut_and_count=cut_and_count, mass_range=mass_range)

        if vars == "all":
            vars_tags = [v for v in all_probe_events.fields if v.startswith("tag_Ele_")]
            vars_probes = [v for v in all_probe_events.fields if v.startswith("ph_")]
            vars = (
                vars_tags
                + vars_probes
                + ["event", "run", "luminosityBlock"]
                + [
                    x
                    for x in all_probe_events.fields
                    if "weight" in x or "Weight" in x or x == "PV_npvs" or x == "Rho_fixedGridRhoAll" or x == "Rho_fixedGridRhoFastjetAll"
                ]
            )
            if all_probe_events.metadata.get("isMC"):
                vars = [*vars, "Pileup_nTrueInt"]

        if cut_and_count:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs)
        else:
            probes = dak.zip({var: all_probe_events[var] for var in vars if "to_use" not in var} | passing_locs | {"pair_mass": all_probe_events["pair_mass"]})

        if all_probe_events.metadata.get("isMC") and "weight" not in vars:
            if "pileupJSON" in all_probe_events.metadata:
                pileup_corr = load_correction(all_probe_events.metadata["pileupJSON"])
            elif "pileupData" in all_probe_events.metadata and "pileupMC" in all_probe_events.metadata:
                pileup_corr = create_correction(all_probe_events.metadata["pileupData"], all_probe_events.metadata["pileupMC"])
            else:
                pileup_corr = None
            if pileup_corr is not None:
                pileup_weight = get_pileup_weight(all_probe_events.Pileup_nTrueInt, pileup_corr)
                probes["weight"] = pileup_weight

        return probes
