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
        trigger_pt=None,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args={},
    ):
        """Tag and Probe efficiency from E/Gamma NTuples

        Parameters
        ----------
            fileset: dict
                The fileset to calculate the trigger efficiencies for.
            filter: str
                The name of the filter to calculate the efficiencies for.
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
        """Get the Pt and Eta arrays of the passing and all probes.
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
        """
        if uproot_options is None:
            uproot_options = {}

        to_compute = apply_to_fileset(
            data_manipulation=self._find_probes,
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
        """Get the Pt and Eta histograms of the passing and all probes.

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
                A dictionary of the form `{"name": [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all], ...}`
                Where each `"name"` is the name of each eta region defined by the user.
                `hpt_pass` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of the passing probes.
                `hpt_all` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of all probes.
                `heta_pass` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of the passing probes.
                `heta_all` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of all probes.
                `hphi_pass` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of the passing probes.
                `hphi_all` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of all probes.
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

        data_manipulation = partial(
            self._make_tnp_histograms,
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

    def _find_probes(self, events):
        if self.goldenjson is not None:
            lumimask = LumiMask(self.goldenjson)
            mask = lumimask(events.run, events.lumi)
            events = events[mask]

        pass_pt_tags = events.tag_Ele_pt > 35
        pass_pt_probes = events.el_pt > self.trigger_pt
        pass_tight_id = events.passingCutBasedTight122XV1 == 1
        in_mass_window = abs(events.pair_mass - 91.1876) < 30
        all_probe_events = events[
            pass_tight_id & in_mass_window & pass_pt_tags & pass_pt_probes
        ]
        passing_probe_events = all_probe_events[all_probe_events[self.filter] == 1]

        passing_probes = dak.zip(
            {
                "pt": passing_probe_events.el_pt,
                "eta": passing_probe_events.el_eta,
                "phi": passing_probe_events.el_phi,
            }
        )
        all_probes = dak.zip(
            {
                "pt": all_probe_events.el_pt,
                "eta": all_probe_events.el_eta,
                "phi": all_probe_events.el_phi,
            }
        )

        return passing_probes, all_probes

    def _make_tnp_histograms(
        self,
        events,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
    ):
        import hist
        from hist.dask import Hist

        import egamma_tnp

        ptbins = egamma_tnp.config.get("ptbins")
        etabins = egamma_tnp.config.get("etabins")
        phibins = egamma_tnp.config.get("phibins")

        passing_probes, all_probes = self._find_probes(events)

        pt_pass = passing_probes.pt
        pt_all = all_probes.pt
        eta_pass = passing_probes.eta
        eta_all = all_probes.eta
        phi_pass = passing_probes.phi
        phi_all = all_probes.phi

        histograms = {}
        histograms["pt"] = {}
        histograms["eta"] = {}
        histograms["phi"] = {}

        plateau_mask_pass = pt_pass > plateau_cut
        plateau_mask_all = pt_all > plateau_cut

        for name_pt, region_pt in eta_regions_pt.items():
            eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (
                abs(eta_pass) < region_pt[1]
            )
            eta_mask_pt_all = (abs(eta_all) > region_pt[0]) & (
                abs(eta_all) < region_pt[1]
            )
            hpt_pass = Hist(
                hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
            )
            hpt_all = Hist(
                hist.axis.Variable(ptbins, name=f"hpt_{name_pt}", label="Pt [GeV]")
            )
            hpt_pass.fill(pt_pass[eta_mask_pt_pass])
            hpt_all.fill(pt_all[eta_mask_pt_all])

            histograms["pt"][name_pt] = {"passing": hpt_pass, "all": hpt_all}

        for name_eta, region_eta in eta_regions_eta.items():
            eta_mask_eta_pass = (abs(eta_pass) > region_eta[0]) & (
                abs(eta_pass) < region_eta[1]
            )
            eta_mask_eta_all = (abs(eta_all) > region_eta[0]) & (
                abs(eta_all) < region_eta[1]
            )
            heta_pass = Hist(
                hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
            )
            heta_all = Hist(
                hist.axis.Variable(etabins, name=f"heta_{name_eta}", label="eta")
            )
            heta_pass.fill(eta_pass[plateau_mask_pass & eta_mask_eta_pass])
            heta_all.fill(eta_all[plateau_mask_all & eta_mask_eta_all])

            histograms["eta"][name_eta] = {"passing": heta_pass, "all": heta_all}

        for name_phi, region_phi in eta_regions_phi.items():
            eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (
                abs(eta_pass) < region_phi[1]
            )
            eta_mask_phi_all = (abs(eta_all) > region_phi[0]) & (
                abs(eta_all) < region_phi[1]
            )
            hphi_pass = Hist(
                hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
            )
            hphi_all = Hist(
                hist.axis.Variable(phibins, name=f"hphi_{name_phi}", label="phi")
            )
            hphi_pass.fill(phi_pass[plateau_mask_pass & eta_mask_phi_pass])
            hphi_all.fill(phi_all[plateau_mask_all & eta_mask_phi_all])

            histograms["phi"][name_phi] = {"passing": hphi_pass, "all": hphi_all}

        return histograms
