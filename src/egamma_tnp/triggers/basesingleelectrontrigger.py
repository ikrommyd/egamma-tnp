import dask_awkward as dak

from egamma_tnp.triggers.basetrigger import BaseTrigger


def _get_arrays(events, perform_tnp, **kwargs):
    p1, a1, p2, a2 = perform_tnp(events, **kwargs)

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


def _get_and_compute_arrays(events, perform_tnp, scheduler, progress, report, **kwargs):
    import dask
    from dask.diagnostics import ProgressBar

    arrays = _get_arrays(events, perform_tnp, **kwargs)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res, report = dask.compute(arrays, report, scheduler=scheduler)

    if progress:
        pbar.unregister()

    return res, report


def _get_tnp_histograms(
    events,
    plateau_cut,
    eta_regions_pt,
    eta_regions_eta,
    eta_regions_phi,
    bins,
    perform_tnp,
    **kwargs,
):
    import hist
    from hist.dask import Hist

    ptbins = bins["ptbins"]
    etabins = bins["etabins"]
    phibins = bins["phibins"]

    arrays = _get_arrays(events, perform_tnp, **kwargs)
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


def _get_and_compute_tnp_histograms(
    events,
    plateau_cut,
    eta_regions_pt,
    eta_regions_eta,
    eta_regions_phi,
    bins,
    perform_tnp,
    scheduler,
    progress,
    report,
    **kwargs,
):
    import dask
    from dask.diagnostics import ProgressBar

    histograms = _get_tnp_histograms(
        events,
        plateau_cut,
        eta_regions_pt,
        eta_regions_eta,
        eta_regions_phi,
        bins,
        perform_tnp,
        **kwargs,
    )

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res, report = dask.compute(histograms, report, scheduler=scheduler)

    if progress:
        pbar.unregister()

    return res, report


class BaseSingleElectronTrigger(BaseTrigger):
    """BaseSingleElectronTrigger class for HLT Trigger efficiency from NanoAOD.

    This class holds the basic methods for all the Tag and Probe classes for different single electron triggers.
    """

    def __init__(
        self,
        names,
        perform_tnp,
        pt,
        avoid_ecal_transition_tags,
        avoid_ecal_transition_probes,
        goldenjson,
        toquery,
        redirector,
        preprocess,
        preprocess_args,
        extra_filter,
        extra_filter_args,
    ):
        super().__init__(
            names,
            perform_tnp,
            avoid_ecal_transition_tags,
            avoid_ecal_transition_probes,
            goldenjson,
            toquery,
            redirector,
            preprocess,
            preprocess_args,
            extra_filter,
            extra_filter_args,
        )
        self.pt = pt

    def get_arrays(self, compute=False, scheduler=None, progress=True):
        """Get the Pt and Eta arrays of the passing and all probes.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            compute : bool, optional
                Whether to return the computed arrays or the delayed arrays.
                The default is False.
            scheduler : str, optional
                The dask scheduler to use. The default is None.
                Only used if compute is True.
            progress : bool, optional
                Whether to show a progress bar if `compute` is True. The default is True.
                Only used if compute is True and no distributed Client is used.

        Returns
        -------
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
        """
        if compute:
            arrays, report = _get_and_compute_arrays(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
                goldenjson=self.goldenjson,
                scheduler=scheduler,
                progress=progress,
                report=self.report,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
        else:
            arrays = _get_arrays(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
                goldenjson=self.goldenjson,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
            report = self.report
        return arrays, report

    def get_tnp_histograms(
        self,
        plateau_cut=None,
        eta_regions_pt=None,
        eta_regions_eta=None,
        eta_regions_phi=None,
        compute=False,
        scheduler=None,
        progress=True,
    ):
        """Get the Pt and Eta histograms of the passing and all probes.

        Parameters
        ----------
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
                Whether to show a progress bar if `compute` is True. The default is True.
                Only used if compute is True and no distributed Client is used.

        Returns
        -------
            histograms : dict
                A dictionary of the form `{"name": [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all], ...}`
                Where each `"name"` is the name of each eta region defined by the user.
                `hpt_pass` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of the passing probes.
                `hpt_all` is a hist.Hist or hist.dask.Hist histogram of the Pt histogram of all probes.
                `heta_pass` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of the passing probes.
                `heta_all` is a hist.Hist or hist.dask.Hist histogram of the Eta histogram of all probes.
                `hphi_pass` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of the passing probes.
                `hphi_all` is a hist.Hist or hist.dask.Hist histogram of the Phi histogram of all probes.
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

        if compute:
            histograms, report = _get_and_compute_tnp_histograms(
                events=self.events,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
                bins=self._bins,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
                goldenjson=self.goldenjson,
                scheduler=scheduler,
                progress=progress,
                report=self.report,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
        else:
            histograms = _get_tnp_histograms(
                events=self.events,
                plateau_cut=plateau_cut,
                eta_regions_pt=eta_regions_pt,
                eta_regions_eta=eta_regions_eta,
                eta_regions_phi=eta_regions_phi,
                bins=self._bins,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                avoid_ecal_transition_tags=self.avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=self.avoid_ecal_transition_probes,
                goldenjson=self.goldenjson,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
            report = self.report
        return histograms, report
