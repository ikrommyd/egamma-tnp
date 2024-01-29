import json
import os
from functools import partial

import dask_awkward as dak
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
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes
        self.goldenjson = goldenjson
        self._extra_filter = extra_filter
        self._extra_filter_args = extra_filter_args

        if goldenjson is not None and not os.path.exists(goldenjson):
            raise FileNotFoundError(f"Golden JSON {goldenjson} does not exist.")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config/runtime_config.json"
        )
        with open(config_path) as f:
            self._bins = json.load(f)

    def get_tnp_arrays(
        self,
        leg="both",
        schemaclass=NanoAODSchema,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt and Eta arrays of the passing and all probes.
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
            arrays : dict of tuples of the same form as fileset where for each dataset the following arrays are present:
            The fileset where for each dataset the following arrays are added:
                pt_pass1: awkward.Array or dask_awkward.Array
                    The Pt array of the passing probes when the firsts electrons are the tags.
                pt_pass2: awkward.Array or dask_awkward.Array
                    The Pt array of the passing probes when the seconds electrons are the tags.
                pt_all1: awkward.Array or dask_awkward.Array
                    The Pt array of all probes when the firsts electrons are the tags.
                pt_all2: awkward.Array or dask_awkward.Array
                    The Pt array of all probes when the seconds electrons are the tags.
                eta_pass1: awkward.Array or dask_awkward.Array
                    The Eta array of the passing probes when the firsts electrons are the tags.
                eta_pass2: awkward.Array or dask_awkward.Array
                    The Eta array of the passing probes when the seconds electrons are the tags.
                eta_all1: awkward.Array or dask_awkward.Array
                    The Eta array of all probes when the firsts electrons are the tags.
                eta_all2: awkward.Array or dask_awkward.Array
                    The Eta array of all probes when the seconds electrons are the tags.
                phi_pass1: awkward.Array or dask_awkward.Array
                    The Phi array of the passing probes when the firsts electrons are the tags.
                phi_pass2: awkward.Array or dask_awkward.Array
                    The Phi array of the passing probes when the seconds electrons are the tags.
                phi_all1: awkward.Array or dask_awkward.Array
                    The Phi array of all probes when the firsts electrons are the tags.
                phi_all2: awkward.Array or dask_awkward.Array
                    The Phi array of all probes when the seconds electrons are the tags.
                report: dict of awkward arrays of the same form as fileset.
                    For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}

        perform_tnp_leg1 = self._tnpimpl_class(
            pt=self.pt1,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg1 = partial(
            self._get_tnp_arrays_on_leg_core, perform_tnp=perform_tnp_leg1, leg="leg1"
        )
        perform_tnp_leg2 = self._tnpimpl_class(
            pt=self.pt2,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg2 = partial(
            self._get_tnp_arrays_on_leg_core, perform_tnp=perform_tnp_leg2, leg="leg2"
        )
        data_manipulation_both = partial(
            self._get_tnp_arrays_on_both_legs_core,
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
        plateau_cut=None,
        eta_regions_pt=None,
        eta_regions_eta=None,
        eta_regions_phi=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """Get the Pt and Eta histograms of the passing and all probes.

        Parameters
        ----------
            leg : str, optional
                The leg to get the histograms for. Can be "first", "second" or "both".
                The default is "both".
            schemaclass: BaseSchema, default NanoAODSchema
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
                A dictionary of the form `{"leg": {"var": {"name": {"passing": passing_probes, "all": all_probes}, ...}, ...}, ...}`
                where "leg" can be "leg1" and/or "leg2" depending on the leg parameter.
                `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
                Each `"name"` is the name of eta region specified by the user and `passing_probes` and `all_probes` are `hist.dask.Hist` objects.
                These are the histograms of the passing and all probes respectively.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.

        """
        if plateau_cut is None:
            plateau_cut = 0
        if eta_regions_pt is None:
            eta_regions_pt = {
                "barrel": [0.0, 1.4442],
                "endcap": [1.566, 2.5],
            }
        if eta_regions_eta is None:
            eta_regions_eta = {"entire": [0.0, 2.5]}
        if eta_regions_phi is None:
            eta_regions_phi = {"entire": [0.0, 2.5]}

        if uproot_options is None:
            uproot_options = {}

        perform_tnp_leg1 = self._tnpimpl_class(
            pt=self.pt1,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg1 = partial(
            self._get_tnp_histograms_on_leg_core,
            perform_tnp=perform_tnp_leg1,
            leg="leg1",
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
            bins=self._bins,
        )
        perform_tnp_leg2 = self._tnpimpl_class(
            pt=self.pt2,
            avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
            avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
            goldenjson=self.goldenjson,
            extra_filter=self._extra_filter,
            extra_filter_args=self._extra_filter_args,
        )
        data_manipulation_leg2 = partial(
            self._get_tnp_histograms_on_leg_core,
            perform_tnp=perform_tnp_leg2,
            leg="leg2",
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
            bins=self._bins,
        )
        data_manipulation_both = partial(
            self._get_tnp_histograms_on_both_legs_core,
            perform_tnp_leg1=perform_tnp_leg1,
            perform_tnp_leg2=perform_tnp_leg2,
            plateau_cut=plateau_cut,
            eta_regions_pt=eta_regions_pt,
            eta_regions_eta=eta_regions_eta,
            eta_regions_phi=eta_regions_phi,
            bins=self._bins,
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

    def _get_tnp_arrays_on_leg(self, events, perform_tnp):
        p1, a1, p2, a2 = perform_tnp(events)

        pt_pass1 = dak.flatten(p1.pt)
        pt_pass2 = dak.flatten(p2.pt)
        pt_all1 = dak.flatten(a1.pt)
        pt_all2 = dak.flatten(a2.pt)

        eta_pass1 = dak.flatten(p1.eta)
        eta_pass2 = dak.flatten(p2.eta)
        eta_all1 = dak.flatten(a1.eta)
        eta_all2 = dak.flatten(a2.eta)

        phi_pass1 = dak.flatten(p1.phi)
        phi_pass2 = dak.flatten(p2.phi)
        phi_all1 = dak.flatten(a1.phi)
        phi_all2 = dak.flatten(a2.phi)

        return (
            pt_pass1,
            pt_pass2,
            pt_all1,
            pt_all2,
            eta_pass1,
            eta_pass2,
            eta_all1,
            eta_all2,
            phi_pass1,
            phi_pass2,
            phi_all1,
            phi_all2,
        )

    def _get_tnp_arrays_on_leg_core(self, events, perform_tnp, leg):
        return {leg: self._get_tnp_arrays_on_leg(events, perform_tnp)}

    def _get_tnp_arrays_on_both_legs_core(
        self, events, perform_tnp_leg1, perform_tnp_leg2
    ):
        return {
            "leg1": self._get_tnp_arrays_on_leg(events, perform_tnp_leg1),
            "leg2": self._get_tnp_arrays_on_leg(events, perform_tnp_leg2),
        }

    def _get_tnp_histograms_on_leg(
        self,
        events,
        perform_tnp,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
        bins,
    ):
        import hist
        from hist.dask import Hist

        ptbins = bins["ptbins"]
        etabins = bins["etabins"]
        phibins = bins["phibins"]

        arrays = self._get_tnp_arrays_on_leg(events, perform_tnp)
        (
            pt_pass1,
            pt_pass2,
            pt_all1,
            pt_all2,
            eta_pass1,
            eta_pass2,
            eta_all1,
            eta_all2,
            phi_pass1,
            phi_pass2,
            phi_all1,
            phi_all2,
        ) = arrays

        histograms = {}
        histograms["pt"] = {}
        histograms["eta"] = {}
        histograms["phi"] = {}

        plateau_mask_pass1 = pt_pass1 > plateau_cut
        plateau_mask_pass2 = pt_pass2 > plateau_cut
        plateau_mask_all1 = pt_all1 > plateau_cut
        plateau_mask_all2 = pt_all2 > plateau_cut

        for name_pt, region_pt in eta_regions_pt.items():
            eta_mask_pt_pass1 = (abs(eta_pass1) > region_pt[0]) & (
                abs(eta_pass1) < region_pt[1]
            )
            eta_mask_pt_pass2 = (abs(eta_pass2) > region_pt[0]) & (
                abs(eta_pass2) < region_pt[1]
            )
            eta_mask_pt_all1 = (abs(eta_all1) > region_pt[0]) & (
                abs(eta_all1) < region_pt[1]
            )
            eta_mask_pt_all2 = (abs(eta_all2) > region_pt[0]) & (
                abs(eta_all2) < region_pt[1]
            )
            hpt_pass = Hist(
                hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
            )
            hpt_all = Hist(
                hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
            )
            hpt_pass.fill(pt_pass1[eta_mask_pt_pass1])
            hpt_pass.fill(pt_pass2[eta_mask_pt_pass2])
            hpt_all.fill(pt_all1[eta_mask_pt_all1])
            hpt_all.fill(pt_all2[eta_mask_pt_all2])

            histograms["pt"][name_pt] = {"passing": hpt_pass, "all": hpt_all}

        for name_eta, region_eta in eta_regions_eta.items():
            eta_mask_eta_pass1 = (abs(eta_pass1) > region_eta[0]) & (
                abs(eta_pass1) < region_eta[1]
            )
            eta_mask_eta_pass2 = (abs(eta_pass2) > region_eta[0]) & (
                abs(eta_pass2) < region_eta[1]
            )
            eta_mask_eta_all1 = (abs(eta_all1) > region_eta[0]) & (
                abs(eta_all1) < region_eta[1]
            )
            eta_mask_eta_all2 = (abs(eta_all2) > region_eta[0]) & (
                abs(eta_all2) < region_eta[1]
            )
            heta_pass = Hist(
                hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
            )
            heta_all = Hist(
                hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
            )
            heta_pass.fill(eta_pass1[plateau_mask_pass1 & eta_mask_eta_pass1])
            heta_pass.fill(eta_pass2[plateau_mask_pass2 & eta_mask_eta_pass2])
            heta_all.fill(eta_all1[plateau_mask_all1 & eta_mask_eta_all1])
            heta_all.fill(eta_all2[plateau_mask_all2 & eta_mask_eta_all2])

            histograms["eta"][name_eta] = {"passing": heta_pass, "all": heta_all}

        for name_phi, region_phi in eta_regions_phi.items():
            eta_mask_phi_pass1 = (abs(eta_pass1) > region_phi[0]) & (
                abs(eta_pass1) < region_phi[1]
            )
            eta_mask_phi_pass2 = (abs(eta_pass2) > region_phi[0]) & (
                abs(eta_pass2) < region_phi[1]
            )
            eta_mask_phi_all1 = (abs(eta_all1) > region_phi[0]) & (
                abs(eta_all1) < region_phi[1]
            )
            eta_mask_phi_all2 = (abs(eta_all2) > region_phi[0]) & (
                abs(eta_all2) < region_phi[1]
            )
            hphi_pass = Hist(
                hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
            )
            hphi_all = Hist(
                hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
            )
            hphi_pass.fill(phi_pass1[plateau_mask_pass1 & eta_mask_phi_pass1])
            hphi_pass.fill(phi_pass2[plateau_mask_pass2 & eta_mask_phi_pass2])
            hphi_all.fill(phi_all1[plateau_mask_all1 & eta_mask_phi_all1])
            hphi_all.fill(phi_all2[plateau_mask_all2 & eta_mask_phi_all2])

            histograms["phi"][name_phi] = {"passing": hphi_pass, "all": hphi_all}

        return histograms

    def _get_tnp_histograms_on_leg_core(
        self,
        events,
        perform_tnp,
        leg,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
        bins,
    ):
        return {
            leg: self._get_tnp_histograms_on_leg(
                events,
                perform_tnp,
                plateau_cut,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
                bins,
            )
        }

    def _get_tnp_histograms_on_both_legs_core(
        self,
        events,
        perform_tnp_leg1,
        perform_tnp_leg2,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
        bins,
    ):
        return {
            "leg1": self._get_tnp_histograms_on_leg(
                events,
                perform_tnp_leg1,
                plateau_cut,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
                bins,
            ),
            "leg2": self._get_tnp_histograms_on_leg(
                events,
                perform_tnp_leg2,
                plateau_cut,
                eta_regions_pt,
                eta_regions_eta,
                eta_regions_phi,
                bins,
            ),
        }
