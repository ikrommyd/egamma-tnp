from __future__ import annotations

import os
from functools import partial

from coffea.dataset_tools import apply_to_fileset


class BaseTagNProbe:
    """Base class for Tag and Probe classes."""

    def __init__(
        self,
        fileset,
        filter,
        tags_pt_cut,
        probes_pt_cut,
        tags_abseta_cut,
        cutbased_id,
        goldenjson,
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
        self.schemaclass = schemaclass
        self.default_vars = default_vars

        if goldenjson is not None and not os.path.exists(goldenjson):
            raise FileNotFoundError(f"Golden JSON {goldenjson} does not exist.")

    def find_probes(self, events, cut_and_count, vars):
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
            vars : list
                The list of variables of the probes to return.

        Returns
        _______
            A tuple of the form (passing_probes, failing_probes) where passing_probes and failing_probes are awkward zip items.
            Each of the zip items has the fields specified in `vars`.
        """
        raise NotImplementedError("find_probes method must be implemented.")

    def get_tnp_arrays(
        self,
        cut_and_count=True,
        vars=None,
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
                The list of variables of the probes to return. The default is ["el_pt", "el_eta", "el_phi"].
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
            arrays: a dictionary of the form {"passing": passing_probes, "failing": failing_probes}
                where passing_probes and failing_probes are awkward zip items.
                Each of the zip items has the fields specified in `vars`.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}
        if vars is None:
            vars = self.default_vars

        def data_manipulation(events):
            passing_probes, failing_probes = self.find_probes(events, cut_and_count=cut_and_count, vars=vars)
            return {"passing": passing_probes, "failing": failing_probes}

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
        cut_and_count=True,
        plateau_cut=None,
        eta_regions_pt=None,
        eta_regions_eta=None,
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
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
            plateau_cut : int or float, optional
                The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
                The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
            eta_regions_pt : dict, optional
                Only used if `pt_eta_phi_1d` is True.
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
        if uproot_options is None:
            uproot_options = {}
        if vars is None:
            vars = self.default_vars

        if cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                pt_eta_phi_1d=True,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
                pt_eta_phi_1d=True,
                vars=vars,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
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
        cut_and_count=True,
        vars=None,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the N-dimensional histograms of the passing and failing probes.

        Parameters
        ----------
            cut_and_count: bool, optional
                Whether to use the cut and count method to find the probes coming from a Z boson.
                If False, invariant mass histograms of the tag-probe pairs will be filled to be fit by a Signal+Background model.
                The default is True.
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
        if uproot_options is None:
            uproot_options = {}
        if vars is None:
            vars = self.default_vars

        if cut_and_count:
            data_manipulation = partial(
                self._make_cutncount_histograms,
                pt_eta_phi_1d=False,
                vars=vars,
                plateau_cut=None,
                eta_regions_pt=None,
                eta_regions_eta=None,
                eta_regions_phi=None,
            )
        else:
            data_manipulation = partial(
                self._make_mll_histograms,
                pt_eta_phi_1d=False,
                vars=vars,
                plateau_cut=None,
                eta_regions_pt=None,
                eta_regions_eta=None,
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

        passing_probes, failing_probes = self.find_probes(events, cut_and_count=True, vars=vars)

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_cutncount_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
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

        passing_probes, failing_probes = self.find_probes(events, cut_and_count=False, vars=vars)

        if pt_eta_phi_1d:
            return fill_pt_eta_phi_mll_histograms(
                passing_probes,
                failing_probes,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
                vars=vars,
            )
        else:
            return fill_nd_mll_histograms(
                passing_probes,
                failing_probes,
                vars=vars,
            )
