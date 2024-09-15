from __future__ import annotations

from functools import partial

import dask_awkward as dak
from coffea.dataset_tools import apply_to_fileset


class BaseTagNProbe:
    """Base class for Tag and Probe classes."""

    def __init__(
        self,
        fileset,
        filters,
        tags_pt_cut,
        probes_pt_cut,
        tags_abseta_cut,
        probes_abseta_cut,
        cutbased_id,
        extra_zcands_mask,
        extra_filter,
        extra_filter_args,
        use_sc_eta,
        use_sc_phi,
        avoid_ecal_transition_tags,
        avoid_ecal_transition_probes,
        schemaclass,
        default_vars,
    ):
        if extra_filter_args is None:
            extra_filter_args = {}
        if filters is not None:
            if probes_pt_cut is None and len(filters) == 1:
                from egamma_tnp.utils.misc import find_pt_threshold

                self.probes_pt_cut = find_pt_threshold(filters[0]) - 3
            elif probes_pt_cut is None and len(filters) > 1:
                self.probes_pt_cut = 5
            else:
                self.probes_pt_cut = probes_pt_cut
        else:
            self.probes_pt_cut = 5
        if not isinstance(filters, list) and filters is not None:
            raise ValueError("filters must be a list of strings or None.")
            if not all(isinstance(f, str) for f in filters):
                raise ValueError("filters must be a list of strings or None.")

        self.fileset = fileset
        self.filters = filters
        self.tags_pt_cut = tags_pt_cut
        self.tags_abseta_cut = tags_abseta_cut
        self.probes_abseta_cut = probes_abseta_cut
        self.cutbased_id = cutbased_id
        self.extra_zcands_mask = extra_zcands_mask
        self.extra_filter = extra_filter
        self.extra_filter_args = extra_filter_args
        self.use_sc_eta = use_sc_eta
        self.use_sc_phi = use_sc_phi
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes
        self.schemaclass = schemaclass
        self.default_vars = default_vars

    def find_probes(self, events, cut_and_count, mass_range, vars):
        """Find the passing and failing probes given some events.

        This method will perform the Tag and Probe method given some events and return the passing and failing probe events
        with the fields specified in `vars`. Probe variables can be accessed using `el_*` and tag variables using `tag_Ele_*`.
        Also other event-level variables and other object variables are available. For instance to get the electron Pt of all electrons
        in NanoAOD, you would have to request `Electron_pt` in `vars`.

        Parameters
        ----------
            events : awkward.Array or dask_awarkward.Array
                events read using coffea.nanoevents.NanoEventsFactory.from_root
            cut_and_count : bool
                Whether to consider all tag-probe pairs 30 GeV away from the Z boson mass as Z bosons
                or to compute the invariant mass of all tag-probe pairs between 50 and 130 GeV
                and add it to the returned probes as a `pair_mass` field.
            mass_range: int or float or tuple of two ints or floats, optional
                The allowed mass range of the tag-probe pairs.
                For cut and count efficiencies, it is a single value representing the mass window around the Z mass.
                For invariant masses to be fit with a Sig+Bkg model, it is a tuple of two values representing the mass range.
            vars : list
                The list of variables of the probes to return.

        Returns
        _______
            probes : awkward.Array or dask_awarkward.Array
                An array with fields specified in `vars` and a boolean field for each filter.
                Also contains a `pair_mass` field if cut_and_count is False.
        """
        raise NotImplementedError("find_probes method must be implemented.")

    def get_tnp_arrays(
        self,
        cut_and_count=True,
        mass_range=None,
        vars=None,
        flat=False,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get arrays of tag, probe and event-level variables.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            mass_range: int or float or tuple of two ints or floats, optional
                The allowed mass range of the tag-probe pairs.
                For cut and count efficiencies, it is a single value representing the mass window around the Z mass.
                For invariant masses to be fit with a Sig+Bkg model, it is a tuple of two values representing the mass range.
                If None, the default is 30 GeV around the Z mass for cut and count efficiencies and 50-130 GeV range otherwise.
                The default is None.
            vars: list, optional
                The list of variables of the probes to return. The default is ["el_pt", "el_eta", "el_phi"].
            flat: bool, optional
                Whether to return flat arrays. The otherwise output needs to be flattenable.
                The default is False.
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
            A tuple of the form (array, report) if `allow_read_errors_with_report` is True, otherwise just arrays.
            arrays: a zip object containing the fields specified in `vars` and a field with a boolean array for each filter.
                It will also contain the `pair_mass` field if cut_and_count is False.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}
        if mass_range is None:
            if cut_and_count:
                mass_range = 30
            else:
                mass_range = (50, 130)
        if cut_and_count and isinstance(mass_range, tuple):
            raise ValueError("For cut and count efficiencies, mass_range must be a single value representing the mass window around the Z mass.")
        if not cut_and_count and not isinstance(mass_range, tuple):
            raise ValueError(
                "For invariant masses to be fit with a Sig+Bkg model, mass_range must be a tuple of two values representing the bounds of the mass range."
            )
        if vars is None:
            vars = self.default_vars

        if flat:
            from egamma_tnp.utils.histogramming import flatten_array

            def data_manipulation(events):
                return flatten_array(self.find_probes(events, cut_and_count=cut_and_count, mass_range=mass_range, vars=vars))
        else:
            data_manipulation = partial(self.find_probes, cut_and_count=cut_and_count, mass_range=mass_range, vars=vars)

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=self.schemaclass,
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

    def get_passing_and_failing_probes(
        self,
        filter,
        cut_and_count=True,
        mass_range=None,
        vars=None,
        flat=False,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the passing and failing probe arrays for a specific filter.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            filter: str
                The filter to check whether the probes pass or not.
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            mass_range: int or float or tuple of two ints or floats, optional
                The allowed mass range of the tag-probe pairs.
                For cut and count efficiencies, it is a single value representing the mass window around the Z mass.
                For invariant masses to be fit with a Sig+Bkg model, it is a tuple of two values representing the mass range.
                If None, the default is 30 GeV around the Z mass for cut and count efficiencies and 50-130 GeV range otherwise.
                The default is None.
            vars: list, optional
                The list of variables of the probes to return. The default is ["el_pt", "el_eta", "el_phi"].
            flat: bool, optional
                Whether to return flat arrays. The otherwise output needs to be flattenable.
                The default is False.
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
            A tuple of the form (array, report) if `allow_read_errors_with_report` is True, otherwise just arrays.
            arrays: a dict of the form {"passing": passing_probes, "failing": failing_probes}
                where passing_probes and failing_probes are  zip objects containing the fields specified in `vars` and a field with a boolean array for each filter.
                It will also contain the `pair_mass` field if cut_and_count is False.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if self.filters is None:
            raise ValueError("filters must be specified during class initialization to use this method.")
        if uproot_options is None:
            uproot_options = {}
        if mass_range is None:
            if cut_and_count:
                mass_range = 30
            else:
                mass_range = (50, 130)
        if cut_and_count and isinstance(mass_range, tuple):
            raise ValueError("For cut and count efficiencies, mass_range must be a single value representing the mass window around the Z mass.")
        if not cut_and_count and not isinstance(mass_range, tuple):
            raise ValueError(
                "For invariant masses to be fit with a Sig+Bkg model, mass_range must be a tuple of two values representing the bounds of the mass range."
            )
        if vars is None:
            vars = self.default_vars

        if flat:
            from egamma_tnp.utils.histogramming import flatten_array

            def data_manipulation(events):
                p_and_f = self._make_passing_and_failing_probes(events, filter, cut_and_count=cut_and_count, mass_range=mass_range, vars=vars)
                return {key: flatten_array(value) for key, value in p_and_f.items()}
        else:
            data_manipulation = partial(
                self._make_passing_and_failing_probes,
                filter=filter,
                cut_and_count=cut_and_count,
                mass_range=mass_range,
                vars=vars,
            )

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=self.schemaclass,
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

    def get_1d_pt_eta_phi_tnp_histograms(
        self,
        filter,
        cut_and_count=True,
        mass_range=None,
        plateau_cut=None,
        eta_regions_pt=None,
        phi_regions_eta=None,
        eta_regions_phi=None,
        vars=None,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt, Eta and Phi histograms of the passing and failing probes.

        Parameters
        ----------
            filter: str
                The filter to check whether the probes pass or not.
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            mass_range: int or float or tuple of two ints or floats, optional
                The allowed mass range of the tag-probe pairs.
                For cut and count efficiencies, it is a single value representing the mass window around the Z mass.
                For invariant masses to be fit with a Sig+Bkg model, it is a tuple of two values representing the mass range.
                If None, the default is 30 GeV around the Z mass for cut and count efficiencies and 50-130 GeV range otherwise.
                The default is None.
            plateau_cut : int or float, optional
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
                The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
            eta_regions_pt : dict, optional
                Only used if `pt_eta_phi_1d` is True.
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Pt histograms will be split into those eta regions.
                The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
            phi_regions_eta : dict, optional
                A dictionary of the form `{"name": [phimin, phimax], ...}`
                where name is the name of the region and phimin and phimax are the absolute phi bounds.
                The Eta histograms will be split into those phi regions.
                The default is to use the entire |phi| < 3.32 region.
            eta_regions_phi : dict, optional
                A dictionary of the form `{"name": [etamin, etamax], ...}`
                where name is the name of the region and etamin and etamax are the absolute eta bounds.
                The Phi histograms will be split into those eta regions.
                The default is to use the entire |eta| < 2.5 region.
            vars : list, optional
                A list of the fields that refer to the Pt, Eta, and Phi of the probes.
                Must be in the order of Pt, Eta, and Phi.
                The default is ["el_pt", "el_eta", "el_phi"].
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
        if self.filters is None:
            raise ValueError("filters must be specified during class initialization to use this method.")
        if uproot_options is None:
            uproot_options = {}
        if mass_range is None:
            if cut_and_count:
                mass_range = 30
            else:
                mass_range = (50, 130)
        if cut_and_count and isinstance(mass_range, tuple):
            raise ValueError("For cut and count efficiencies, mass_range must be a single value representing the mass window around the Z mass.")
        if not cut_and_count and not isinstance(mass_range, tuple):
            raise ValueError(
                "For invariant masses to be fit with a Sig+Bkg model, mass_range must be a tuple of two values representing the bounds of the mass range."
            )
        if vars is None:
            vars = self.default_vars

        if cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                filter=filter,
                pt_eta_phi_1d=True,
                mass_range=mass_range,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                phi_regions_eta=phi_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
                filter=filter,
                pt_eta_phi_1d=True,
                mass_range=mass_range,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                phi_regions_eta=phi_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=self.schemaclass,
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

    def get_nd_tnp_histograms(
        self,
        filter,
        cut_and_count=True,
        mass_range=None,
        vars=None,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the N-dimensional histograms of the passing and failing probes.

        Parameters
        ----------
            filter: str
                The filter to check whether the probes pass or not.
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            mass_range: int or float or tuple of two ints or floats, optional
                The allowed mass range of the tag-probe pairs.
                For cut and count efficiencies, it is a single value representing the mass window around the Z mass.
                For invariant masses to be fit with a Sig+Bkg model, it is a tuple of two values representing the mass range.
                If None, the default is 30 GeV around the Z mass for cut and count efficiencies and 50-130 GeV range otherwise.
                The default is None.
            vars: list, optional
                The variables to use to fill the N-dimensional histograms. The default is ["el_pt", "el_eta", "el_phi"].
                These vars will be used to fill the N-dimensional histograms.
                If cut_and_count is False, one more invariant mass axis will be added to the histograms.
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
            histograms: a dictionary of the form {"passing": passing_probes, "failing": failing_probes}
            where passing_probes and failing_probes are hist.Hist or hist.dask.Hist objects.
            These are the N-dimensional histograms of the passing and failing probes respectively.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if self.filters is None:
            raise ValueError("filters must be specified during class initialization to use this method.")
        if uproot_options is None:
            uproot_options = {}
        if mass_range is None:
            if cut_and_count:
                mass_range = 30
            else:
                mass_range = (50, 130)
        if cut_and_count and isinstance(mass_range, tuple):
            raise ValueError("For cut and count efficiencies, mass_range must be a single value representing the mass window around the Z mass.")
        if not cut_and_count and not isinstance(mass_range, tuple):
            raise ValueError(
                "For invariant masses to be fit with a Sig+Bkg model, mass_range must be a tuple of two values representing the bounds of the mass range."
            )
        if vars is None:
            vars = self.default_vars

        if cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                filter=filter,
                pt_eta_phi_1d=False,
                mass_range=mass_range,
                vars=vars,
                plateau_cut=None,
                eta_regions_pt=None,
                phi_regions_eta=None,
                eta_regions_phi=None,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
                filter=filter,
                pt_eta_phi_1d=False,
                mass_range=mass_range,
                vars=vars,
                plateau_cut=None,
                eta_regions_pt=None,
                phi_regions_eta=None,
                eta_regions_phi=None,
            )

        to_compute = apply_to_fileset(
            data_manipulation=data_manipulation,
            fileset=self.fileset,
            schemaclass=self.schemaclass,
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

    def _make_passing_and_failing_probes(self, events, filter, cut_and_count, mass_range, vars):
        probes = self.find_probes(events, cut_and_count=cut_and_count, mass_range=mass_range, vars=vars)
        if "NanoAOD" in self.__class__.__name__:
            has_passing_probe = dak.any(probes[filter], axis=1)
            has_failing_probe = dak.any(~probes[filter], axis=1)
            passing_probes = probes[has_passing_probe]
            failing_probes = probes[has_failing_probe]
            for var in vars:
                if var.startswith(("el_", "tag_Ele_", "ph_")):
                    passing_probes[var] = passing_probes[var][passing_probes[filter]]
                    failing_probes[var] = failing_probes[var][~failing_probes[filter]]
            if "pair_mass" in probes.fields:
                passing_probes["pair_mass"] = passing_probes["pair_mass"][passing_probes[filter]]
                failing_probes["pair_mass"] = failing_probes["pair_mass"][~failing_probes[filter]]
            passing_probes = passing_probes[[x for x in passing_probes.fields if x not in self.filters]]
            failing_probes = failing_probes[[x for x in failing_probes.fields if x not in self.filters]]
        else:
            passing_probes = probes[probes[filter]][[x for x in probes.fields if x not in self.filters]]
            failing_probes = probes[~probes[filter]][[x for x in probes.fields if x not in self.filters]]

        return {"passing": passing_probes, "failing": failing_probes}

    def _make_cutncount_histograms(
        self,
        events,
        filter,
        pt_eta_phi_1d,
        mass_range,
        vars,
        plateau_cut,
        eta_regions_pt,
        phi_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import (
            fill_nd_cutncount_histograms,
            fill_pt_eta_phi_cutncount_histograms,
        )

        p_and_f = self._make_passing_and_failing_probes(events, filter, cut_and_count=True, mass_range=mass_range, vars=vars)
        passing_probes = p_and_f["passing"]
        failing_probes = p_and_f["failing"]

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_cutncount_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                phi_regions_eta=phi_regions_eta,
                eta_regions_phi=eta_regions_phi,
                vars=vars,
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
        filter,
        pt_eta_phi_1d,
        mass_range,
        vars,
        plateau_cut,
        eta_regions_pt,
        phi_regions_eta,
        eta_regions_phi,
    ):
        from egamma_tnp.utils import (
            fill_nd_mll_histograms,
            fill_pt_eta_phi_mll_histograms,
        )

        p_and_f = self._make_passing_and_failing_probes(events, filter, cut_and_count=False, mass_range=mass_range, vars=vars)
        passing_probes = p_and_f["passing"]
        failing_probes = p_and_f["failing"]

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_mll_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                phi_regions_eta=phi_regions_eta,
                eta_regions_phi=eta_regions_phi,
                vars=vars,
            )
        else:
            return fill_nd_mll_histograms(
                passing_probes,
                failing_probes,
                vars=vars,
            )
