from functools import partial

import dask_awkward as dak
from coffea.dataset_tools import apply_to_fileset
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import BaseSchema


class TagNProbeFromNTuples:
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
        if extra_filter_args is None:
            extra_filter_args = {}
        if probes_pt_cut is None:
            from egamma_tnp.utils.misc import find_pt_threshold

            self.probes_pt_cut = find_pt_threshold(filter) - 3
        else:
            self.probes_pt_cut = probes_pt_cut

        self.fileset = fileset
        self.filter = filter
        self.tags_pt_cut = tags_pt_cut
        self.tags_abseta_cut = tags_abseta_cut
        self.cutbased_id = cutbased_id
        self.goldenjson = goldenjson
        self.extra_filter = extra_filter
        self.extra_filter_args = extra_filter_args
        self.use_sc_eta = use_sc_eta
        self.use_sc_phi = use_sc_phi
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"TagNProbeFromNTuples({self.filter}, Number of files: {n_of_files}, Golden JSON: {self.goldenjson})"

    def get_tnp_arrays(
        self,
        cut_and_count=True,
        vars=None,
        schemaclass=BaseSchema,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi arrays of the passing and failing probes.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            vars: list, optional
                The list of variables of the probes to return. The default is ["pt", "eta", "phi"].
            schemaclass: BaseSchema, default BaseSchema
                The nanoevents schema to interpret the input dataset with.
            uproot_options : dict, optional
                Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
            compute : bool, optional
                Whether to return the computed arrays or the delayed arrays.
                The default is False.
            scheduler : str, optional
                The dask scheduler to use. The default is None.
                Only used if compute is True.
            progress : bool, optional
                Whether to show a progress bar if `compute` is True. The default is False.
                Only meaningful if compute is True and no distributed Client is used.

        Returns
        -------
            A tuple of the form (arrays, report) if `allow_read_errors_with_report` is True, otherwise just arrays.
            arrays :a tuple of dask awkward zip items of the form (passing_probes, failing_probes).
                Each of the zip items has the following fields:
                    pt: dask_awkward.Array
                        The Pt array of the probes.
                    eta: dask_awkward.array
                        The Eta array of the probes.
                    phi: dask_awkward.array
                        The Phi array of the probes.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}

        data_manipulation = partial(
            self._find_probes, cut_and_count=cut_and_count, vars=vars
        )

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=schemaclass,
            uproot_options=uproot_options,
        )
        if compute:
            import dask
            from dask.diagnostics import ProgressBar

            if progress:
                pbar = ProgressBar()
                pbar.register()

            computed = dask.compute(to_compute, scheduler=scheduler)

            if progress:
                pbar.unregister()

            return computed[0]

        return to_compute

    def get_tnp_histograms(
        self,
        cut_and_count=True,
        pt_eta_phi_1d=True,
        vars=None,
        plateau_cut=None,
        eta_regions_pt=None,
        eta_regions_eta=None,
        eta_regions_phi=None,
        schemaclass=BaseSchema,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi histograms of the passing and failing probes.

        Parameters
        ----------
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            pt_eta_phi_1d: bool, optional
                Whether to fill 1D Pt, Eta and Phi histograms or N-dimensional histograms. The default is True.
            vars: list, optional
                The list of variables to fill the N-dimensional histograms with.
                The default is ["pt", "eta", "phi"].
            plateau_cut : int or float, optional
                Only used if `pt_eta_phi_1d` is True.
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
                The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
            eta_regions_pt : dict, optional
                Only used if `pt_eta_phi_1d` is True.
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Pt histograms will be split into those eta regions.
                The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
            eta_regions_eta : dict, optional
                Only used if `pt_eta_phi_1d` is True.
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Eta histograms will be split into those eta regions.
                The default is to use the entire |eta| < 2.5 region.
            eta_regions_phi : dict, optional
                Only used if `pt_eta_phi_1d` is True.
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Phi histograms will be split into those eta regions.
                The default is to use the entire |eta| < 2.5 region.
            schemaclass: BaseSchema, default BaseSchema
                 The nanoevents schema to interpret the input dataset with.
            uproot_options : dict, optional
                Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
            compute : bool, optional
                Whether to return the computed hist.Hist histograms or the delayed hist.dask.Hist histograms.
                The default is False.
            scheduler : str, optional
                The dask scheduler to use. The default is None.
                Only used if compute is True.
            progress : bool, optional
                Whether to show a progress bar if `compute` is True. The default is False.
                Only meaningful if compute is True and no distributed Client is used.

        Returns
        -------
            A tuple of the form (histograms, report) if `allow_read_errors_with_report` is True, otherwise just histograms.
            histograms : dict of dicts of the same form as fileset where for each dataset the following dictionary is present:
                A dictionary of the form `{"var": {"name": {"passing": passing_probes, "failing": failing_probes}, ...}, ...}`
                where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
                Each `"name"` is the name of eta region specified by the user and `passing_probes` and `failing_probes` are `hist.dask.Hist` objects.
                These are the histograms of the passing and failing probes respectively.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}

        if cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                pt_eta_phi_1d=pt_eta_phi_1d,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
                pt_eta_phi_1d=pt_eta_phi_1d,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=schemaclass,
            uproot_options=uproot_options,
        )
        if compute:
            import dask
            from dask.diagnostics import ProgressBar

            if progress:
                pbar = ProgressBar()
                pbar.register()

            computed = dask.compute(to_compute, scheduler=scheduler)

            if progress:
                pbar.unregister()

            return computed[0]

        return to_compute

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
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.lumi)
            events = events[mask]
        if self.use_sc_eta:
            events["el_eta"] = events.el_sc_eta
        if self.use_sc_phi:
            events["el_phi"] = events.el_sc_phi

        if self.avoid_ecal_transition_tags:
            pass_eta_ebeegap_tags = (abs(events.tag_Ele_eta) < 1.4442) | (
                abs(events.tag_Ele_eta) > 1.566
            )
            events = events[pass_eta_ebeegap_tags]
        if self.avoid_ecal_transition_probes:
            pass_eta_ebeegap_probes = (abs(events.el_eta) < 1.4442) | (
                abs(events.el_eta) > 1.566
            )
            events = events[pass_eta_ebeegap_probes]

        pass_pt_tags = events.tag_Ele_pt > self.tags_pt_cut
        pass_abseta_tags = abs(events.tag_Ele_eta) < self.tags_abseta_cut
        opposite_charge = events.tag_Ele_q * events.el_q == -1
        events = events[pass_pt_tags & pass_abseta_tags & opposite_charge]

        passing_probe_events, failing_probe_events = self._find_probe_events(
            events, cut_and_count=cut_and_count
        )

        if cut_and_count:
            passing_probes = dak.zip(
                {f"{var}": passing_probe_events[f"el_{var}"] for var in vars}
            )
            failing_probes = dak.zip(
                {f"{var}": failing_probe_events[f"el_{var}"] for var in vars}
            )
        else:
            p_arrays = {f"{var}": passing_probe_events[f"el_{var}"] for var in vars}
            p_arrays["pair_mass"] = passing_probe_events["pair_mass"]
            f_arrays = {f"{var}": failing_probe_events[f"el_{var}"] for var in vars}
            f_arrays["pair_mass"] = failing_probe_events["pair_mass"]
            passing_probes = dak.zip(p_arrays)
            failing_probes = dak.zip(f_arrays)

        return passing_probes, failing_probes

    def _make_cutncount_histograms(
        self,
        events,
        pt_eta_phi_1d,
        vars,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import (
            fill_nd_cutncount_histograms,
            fill_pt_eta_phi_cutncount_histograms,
        )

        passing_probes, failing_probes = self._find_probes(
            events, cut_and_count=True, vars=vars
        )

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_cutncount_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            return fill_nd_cutncount_histograms(
                passing_probes,
                failing_probes,
                vars=vars,
            )

    def _make_mll_histograms(
        self,
        events,
        pt_eta_phi_1d,
        vars,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import (
            fill_nd_mll_histograms,
            fill_pt_eta_phi_mll_histograms,
        )

        passing_probes, failing_probes = self._find_probes(
            events, cut_and_count=False, vars=vars
        )

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_mll_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            return fill_nd_mll_histograms(
                passing_probes,
                failing_probes,
                vars=vars,
            )
