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
        cut_and_count=True,
        cutbased_id=None,
        trigger_pt=None,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        """Tag and Probe efficiency from E/Gamma NTuples

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filter: str
                The name of the filter to calculate the efficiencies for.
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            cutbased_id: str, optional
                The name of the cutbased ID to apply to the probes.
                If None, no cutbased ID is applied. The default is None.
            trigger_pt: int or float, optional
                The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
                Should be very slightly below the Pt threshold of the filter.
                If it is None, it will attempt to infer it from the filter name.
                If it fails to do so, it will set it to 0.
            goldenjson: str, optional
                The golden json to use for luminosity masking. The default is None.
            extra_filter : Callable, optional
                An extra function to filter the events. The default is None.
                Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
            extra_filter_args : dict, optional
                Extra arguments to pass to extra_filter. The default is {}.
        """
        if extra_filter_args is None:
            extra_filter_args = {}
        if trigger_pt is None:
            from egamma_tnp.utils.misc import find_pt_threshold

            self.trigger_pt = find_pt_threshold(filter) - 3
        else:
            self.trigger_pt = trigger_pt
        self.fileset = fileset
        self.filter = filter
        self.cut_and_count = cut_and_count
        self.cutbased_id = cutbased_id
        self.goldenjson = goldenjson
        self.extra_filter = extra_filter
        self.extra_filter_args = extra_filter_args

    def get_tnp_arrays(
        self,
        schemaclass=BaseSchema,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi arrays of the passing and all probes.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
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
            arrays :a tuple of dask awkward zip items of the form (passing_probes, all_probes).
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

        data_manipulation = self._find_probes

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
        schemaclass=BaseSchema,
        uproot_options=None,
        plateau_cut=None,
        eta_regions_pt=None,
        eta_regions_eta=None,
        eta_regions_phi=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi histograms of the passing and all probes.

        Parameters
        ----------
            schemaclass: BaseSchema, default BaseSchema
                 The nanoevents schema to interpret the input dataset with.
            uproot_options : dict, optional
                Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
            plateau_cut : int or float, optional
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
                The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
            eta_regions_pt : dict, optional
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Pt histograms will be split into those eta regions.
                The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
            eta_regions_eta : dict, optional
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Eta histograms will be split into those eta regions.
                The default is to use the entire |eta| < 2.5 region.
            eta_regions_phi : dict, optional
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Phi histograms will be split into those eta regions.
                The default is to use the entire |eta| < 2.5 region.
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
                A dictionary of the form `{"var": {"name": {"passing": passing_probes, "all": all_probes}, ...}, ...}`
                where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
                Each `"name"` is the name of eta region specified by the user and `passing_probes` and `all_probes` are `hist.dask.Hist` objects.
                These are the histograms of the passing and all probes respectively.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}

        if self.cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
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

    def _find_probe_events(self, events):
        pass_pt_tags = events.tag_Ele_pt > 35
        pass_pt_probes = events.el_pt > self.trigger_pt
        if self.cutbased_id:
            pass_cutbased_id = events[self.cutbased_id] == 1
        else:
            pass_cutbased_id = True
        if self.cut_and_count:
            in_mass_window = abs(events.pair_mass - 91.1876) < 30
        else:
            in_mass_window = True
        all_probe_events = events[
            pass_cutbased_id & in_mass_window & pass_pt_tags & pass_pt_probes
        ]
        passing_probe_events = all_probe_events[all_probe_events[self.filter] == 1]

        return passing_probe_events, all_probe_events

    def _find_probes(self, events):
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        passint_probe_events, all_probe_events = self._find_probe_events(events)

        if self.cut_and_count:
            passing_probes = dak.zip(
                {
                    "pt": passint_probe_events.el_pt,
                    "eta": passint_probe_events.el_eta,
                    "phi": passint_probe_events.el_phi,
                }
            )
            all_probes = dak.zip(
                {
                    "pt": all_probe_events.el_pt,
                    "eta": all_probe_events.el_eta,
                    "phi": all_probe_events.el_phi,
                }
            )
        else:
            passing_probes = dak.zip(
                {
                    "pt": passint_probe_events.el_pt,
                    "eta": passint_probe_events.el_eta,
                    "phi": passint_probe_events.el_phi,
                    "pair_mass": passint_probe_events.pair_mass,
                }
            )
            all_probes = dak.zip(
                {
                    "pt": all_probe_events.el_pt,
                    "eta": all_probe_events.el_eta,
                    "phi": all_probe_events.el_phi,
                    "pair_mass": all_probe_events.pair_mass,
                }
            )

        return passing_probes, all_probes

    def _make_cutncount_histograms(
        self,
        events,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import fill_cutncount_histograms

        passing_probes, all_probes = self._find_probes(events)
        return fill_cutncount_histograms(
            passing_probes,
            all_probes,
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )

    def _make_mll_histograms(
        self,
        events,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import fill_mll_histograms

        passing_probes, all_probes = self._find_probes(events)
        return fill_mll_histograms(
            passing_probes,
            all_probes,
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )
