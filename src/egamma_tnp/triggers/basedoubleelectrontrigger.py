import os
from functools import partial

from coffea.dataset_tools import apply_to_fileset
from coffea.nanoevents import NanoAODSchema


class BaseDoubleElectronTrigger:
    """BaseDoubleElectronTrigger class for HLT Trigger efficiency from NanoAOD.

    This class holds the basic methods for all the Tag and Probe classes for different double triggers.
    """

    def __init__(
        self,
        fileset,
        tnpimpl_class,
        pt1,
        pt2,
        filterbit1,
        filterbit2,
        avoid_ecal_transition_tags,
        avoid_ecal_transition_probes,
        goldenjson,
        extra_filter,
        extra_filter_args,
    ):
        self.fileset = fileset
        self._tnpimpl_class = tnpimpl_class
        self.pt1 = pt1
        self.pt2 = pt2
        self.filterbit1 = filterbit1
        self.filterbit2 = filterbit2
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes
        self.goldenjson = goldenjson
        self._extra_filter = extra_filter
        self._extra_filter_args = extra_filter_args

        if goldenjson is not None and not os.path.exists(goldenjson):
            raise FileNotFoundError(f"Golden JSON {goldenjson} does not exist.")

    def get_tnp_arrays(
        self,
        leg="both",
        schemaclass=NanoAODSchema,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi arrays of the passing and all probes.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            leg : str, optional
                The leg to get the arrays for. Can be "first", "second" or "both".
                The default is "both".
            schemaclass: BaseSchema, default NanoAODSchema
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
            array_dict: dict
            A dictionary with keys "leg1" and/or "leg2" depending on the leg parameter.
            The value for each key is a tuple of the form (arrays, report) if `allow_read_errors_with_report` is True, otherwise just arrays.
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

        perform_tnp_leg1 = self._tnpimpl_class(
            pt=self.pt1,
            filterbit=self.filterbit1,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg1 = partial(
            self._make_tnp_arrays_on_leg, perform_tnp=perform_tnp_leg1, leg="leg1"
        )
        perform_tnp_leg2 = self._tnpimpl_class(
            pt=self.pt2,
            filterbit=self.filterbit2,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg2 = partial(
            self._make_tnp_arrays_on_leg, perform_tnp=perform_tnp_leg2, leg="leg2"
        )
        data_manipulation_both = partial(
            self._make_tnp_arrays_on_both_legs,
            perform_tnp_leg1=perform_tnp_leg1,
            perform_tnp_leg2=perform_tnp_leg2,
        )

        if leg == "first":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_leg1,
                fileset=self.fileset,
                schemaclass=schemaclass,
                uproot_options=uproot_options,
            )
        elif leg == "second":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_leg2,
                fileset=self.fileset,
                schemaclass=schemaclass,
                uproot_options=uproot_options,
            )
        elif leg == "both":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_both,
                fileset=self.fileset,
                schemaclass=schemaclass,
                uproot_options=uproot_options,
            )
        else:
            raise ValueError(f"leg must be 'first', 'second' or 'both', not {leg}")

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
        leg="both",
        schemaclass=NanoAODSchema,
        uproot_options=None,
        plateau_cut1=None,
        plateau_cut2=None,
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
            leg : str, optional
                The leg to get the histograms for. Can be "first", "second" or "both".
                The default is "both".
            schemaclass: BaseSchema, default NanoAODSchema
                The nanoevents schema to interpret the input dataset with.
            uproot_options : dict, optional
                Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
            plateau_cut1 : int or float, optional
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms for the first leg.
                The default is None, meaning that no extra cut is applied and the activation region is included in those histograms.
            plateau_cut2 : int or float, optional,
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms for the second leg.
                The default is None, meaning that no extra cut is applied and the activation region is included in those histograms.
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
                A dictionary of the form `{"leg": {"var": {"name": {"passing": passing_probes, "all": all_probes}, ...}, ...}, ...}`
                where "leg" can be "leg1" and/or "leg2" depending on the leg parameter.
                `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
                Each `"name"` is the name of eta region specified by the user and `passing_probes` and `all_probes` are `hist.dask.Hist` objects.
                These are the histograms of the passing and all probes respectively.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.

        """
        if uproot_options is None:
            uproot_options = {}

        perform_tnp_leg1 = self._tnpimpl_class(
            pt=self.pt1,
            filterbit=self.filterbit1,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg1 = partial(
            self._make_tnp_histograms_on_leg,
            perform_tnp=perform_tnp_leg1,
            leg="leg1",
            plateau_cut=plateau_cut1,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )
        perform_tnp_leg2 = self._tnpimpl_class(
            pt=self.pt2,
            filterbit=self.filterbit2,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg2 = partial(
            self._make_tnp_histograms_on_leg,
            perform_tnp=perform_tnp_leg2,
            leg="leg2",
            plateau_cut=plateau_cut2,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )
        data_manipulation_both = partial(
            self._make_tnp_histograms_on_both_legs,
            perform_tnp_leg1=perform_tnp_leg1,
            perform_tnp_leg2=perform_tnp_leg2,
            plateau_cut1=plateau_cut1,
            plateau_cut2=plateau_cut2,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )

        if leg == "first":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_leg1,
                fileset=self.fileset,
                schemaclass=schemaclass,
                uproot_options=uproot_options,
            )
        elif leg == "second":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_leg2,
                fileset=self.fileset,
                schemaclass=schemaclass,
                uproot_options=uproot_options,
            )
        elif leg == "both":
            to_compute = apply_to_fileset(
                data_manipulation=data_manipulation_both,
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

    def _make_tnp_arrays_on_leg(self, events, perform_tnp, leg):
        return {leg: perform_tnp(events)}

    def _make_tnp_arrays_on_both_legs(self, events, perform_tnp_leg1, perform_tnp_leg2):
        return {
            "leg1": perform_tnp_leg1(events),
            "leg2": perform_tnp_leg2(events),
        }

    def _make_tnp_histograms_on_leg_core(
        self,
        events,
        perform_tnp,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import fill_cutncount_histograms

        passing_probes, all_probes = perform_tnp(events)
        return fill_cutncount_histograms(
            passing_probes,
            all_probes,
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
        )

    def _make_tnp_histograms_on_leg(
        self,
        events,
        perform_tnp,
        leg,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        return {
            leg: self._make_tnp_histograms_on_leg_core(
                events,
                perform_tnp,
                plateau_cut,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
            )
        }

    def _make_tnp_histograms_on_both_legs(
        self,
        events,
        perform_tnp_leg1,
        perform_tnp_leg2,
        plateau_cut1,
        plateau_cut2,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        return {
            "leg1": self._make_tnp_histograms_on_leg_core(
                events,
                perform_tnp_leg1,
                plateau_cut1,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
            ),
            "leg2": self._make_tnp_histograms_on_leg_core(
                events,
                perform_tnp_leg2,
                plateau_cut2,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
            ),
        }
