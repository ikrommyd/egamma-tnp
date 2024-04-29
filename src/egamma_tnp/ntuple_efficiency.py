import dask_awkward as dak
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import BaseSchema

from egamma_tnp._base_tagnprobe import BaseTagNProbe


class TagNProbeFromNTuples(BaseTagNProbe):
    def __init__(
        self,
        fileset,
        filter,
        *,
        tags_pt_cut=35,
        probes_pt_cut=None,
        tags_abseta_cut=2.5,
        cutbased_id=None,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
    ):
        """Tag and Probe efficiency from E/Gamma NTuples

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filter: str
                The name of the filter to calculate the efficiencies for.
            tags_pt_cut: int or float, optional
                The Pt cut to apply to the tag electrons. The default is 35.
            probes_pt_cut: int or float, optional
                The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
                Should be very slightly below the Pt threshold of the filter.
                If it is None, it will attempt to infer it from the filter name.
                If it fails to do so, it will set it to 0.
            tags_abseta_cut: int or float, optional
                The absolute Eta cut to apply to the tag electrons. The default is 2.5.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
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
        """
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
            schemaclass=BaseSchema,
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"TagNProbeFromNTuples({self.filter}, Number of files: {n_of_files}, Golden JSON: {self.goldenjson})"

    def _find_probe_events(self, events, cut_and_count):
        pass_pt_probes = events.el_pt > self.probes_pt_cut
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if cut_and_count:
            in_mass_window = abs(events.pair_mass - 91.1876) < 30
        else:
            in_mass_window = (events.pair_mass > 50) & (events.pair_mass < 130)
        all_probe_events = events[pass_cutbased_id & in_mass_window & pass_pt_probes]
        passing_locs = all_probe_events[self.filter] == 1
        passing_probe_events = all_probe_events[passing_locs]
        failing_probe_events = all_probe_events[~passing_locs]

        return passing_probe_events, failing_probe_events

    def _find_probes(self, events, cut_and_count, vars):
        if vars is None:
            vars = ["pt", "eta", "phi"]
        if self.use_sc_eta:
            events["el_eta"] = events.el_sc_eta
        if self.use_sc_phi:
            events["el_phi"] = events.el_sc_phi
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta) < 1.4442) | (abs(events.tag_Ele_eta) > 1.566)
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.el_eta) < 1.4442) | (abs(events.el_eta) > 1.566)
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta) < self.tags_abseta_cut
        opposite_charge = events.tag_Ele_q * events.el_q == -1
        events = events[pass_pt_tags & pass_abseta_tags & opposite_charge]

        passing_probe_events, failing_probe_events = self._find_probe_events(events, cut_and_count=cut_and_count)

        if cut_and_count:
            passing_probes = dak.zip({f"{var}": passing_probe_events[f"el_{var}"] for var in vars})
            failing_probes = dak.zip({f"{var}": failing_probe_events[f"el_{var}"] for var in vars})
        else:
            p_arrays = {f"{var}": passing_probe_events[f"el_{var}"] for var in vars}
            p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
            f_arrays = {f"{var}": failing_probe_events[f"el_{var}"] for var in vars}
            f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
            passing_probes = dak.zip(p_arrays)
            failing_probes = dak.zip(f_arrays)

        return passing_probes, failing_probes
