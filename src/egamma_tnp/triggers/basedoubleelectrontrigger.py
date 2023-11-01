import json
import os

import dask_awkward as dak

from egamma_tnp.utils.dataset import get_nanoevents_file


def _get_arrays_on_leg(events, perform_tnp, **kwargs):
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


def _get_and_compute_arrays_on_leg(events, perform_tnp, scheduler, progress, **kwargs):
    import dask
    from dask.diagnostics import ProgressBar

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
    ) = _get_arrays_on_leg(events, perform_tnp, **kwargs)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res = dask.compute(
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
        scheduler=scheduler,
    )

    if progress:
        pbar.unregister()

    return res


def _get_tnp_histograms_on_leg(
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
    ) = _get_arrays_on_leg(events, perform_tnp, **kwargs)

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


def _get_and_compute_tnp_histograms_on_leg(
    events,
    plateau_cut,
    eta_regions_pt,
    eta_regions_eta,
    eta_regions_phi,
    bins,
    perform_tnp,
    scheduler,
    progress,
    **kwargs,
):
    import dask
    from dask.diagnostics import ProgressBar

    histograms = _get_tnp_histograms_on_leg(
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

    res = dask.compute(histograms, scheduler=scheduler)[0]

    if progress:
        pbar.unregister()

    return res


class BaseDoubleElectronTrigger:
    """BaseDoubleElectronTrigger class for HLT Trigger efficiency from NanoAOD.

    This class holds the basic methods for all the Tag and Probe classes for different double electron triggers.
    """

    def __init__(
        self,
        names,
        perform_tnp,
        pt1,
        pt2,
        avoid_ecal_transition_tags,
        avoid_ecal_transition_probes,
        goldenjson,
        toquery,
        redirect,
        custom_redirector,
        invalid,
        preprocess,
        preprocess_args,
        extra_filter,
        extra_filter_args,
    ):
        self.names = names
        self._perform_tnp = perform_tnp
        self.pt1 = pt1
        self.pt2 = pt2
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes
        self.goldenjson = goldenjson
        self.events = None
        self._toquery = toquery
        self._redirect = redirect
        self._custom_redirector = custom_redirector
        self._invalid = invalid
        self._preprocess = preprocess
        self._preprocess_args = preprocess_args
        self._extra_filter = extra_filter
        self._extra_filter_args = extra_filter_args

        self.file = get_nanoevents_file(
            self.names,
            toquery=self._toquery,
            redirect=self._redirect,
            custom_redirector=self._custom_redirector,
            invalid=self._invalid,
            preprocess=self._preprocess,
            preprocess_args=self._preprocess_args,
        )

        if goldenjson is not None and not os.path.exists(goldenjson):
            raise FileNotFoundError(f"Golden JSON {goldenjson} does not exist.")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config/runtime_config.json"
        )
        with open(config_path) as f:
            self._bins = json.load(f)

    def remove_bad_xrootd_files(self, keys):
        """Remove bad xrootd files from self.file.

        Parameters
        ----------
            keys : str or list of str
                The keys of self.file to remove.
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                self.file.pop(key)
            except KeyError:
                pass

    def redirect_files(self, keys, redirectors="root://cmsxrootd.fnal.gov/"):
        """Redirect the files in self.file.

        Parameters
        ----------
            keys : str or list of str
                The keys of self.file to redirect.
            redirectors : str or list of str, optional
                The redirectors to use. The default is "root://cmsxrootd.fnal.gov/".
                If multiple keys are given, then either one redirector or the same number of redirectors as keys must be given.
        """
        from egamma_tnp.utils import redirect_files

        if isinstance(keys, str):
            keys = [keys]
        if isinstance(redirectors, str) or (
            isinstance(redirectors, list) and len(redirectors) == 1
        ):
            redirectors = (
                [redirectors] * len(keys)
                if isinstance(redirectors, str)
                else redirectors * len(keys)
            )
        if (len(keys) > 1 and (len(redirectors) != 1)) and (
            len(keys) != len(redirectors)
        ):
            raise ValueError(
                f"If multiple keys are given, then either one redirector or the same number of redirectors as keys must be given."
                f"Got {len(keys)} keys and {len(redirectors)} redirectors."
            )
        for key, redirector in zip(keys, redirectors):
            isrucio = True if key[:7] == "root://" else False
            newkey = redirect_files(key, redirector=redirector, isrucio=isrucio).pop()
            self.file[newkey] = self.file.pop(key)

    def load_events(self, from_root_args=None):
        """Load the events from the names.

        Parameters
        ----------
            from_root_args : dict, optional
                Extra arguments to pass to coffea.nanoevents.NanoEventsFactory.from_root().
                The default is {}.
        """
        from coffea.nanoevents import NanoEventsFactory

        if from_root_args is None:
            from_root_args = {}

        self.events = NanoEventsFactory.from_root(
            self.file,
            permit_dask=True,
            **from_root_args,
        ).events()

    def get_arrays(self, leg="both", compute=False, scheduler=None, progress=True):
        """Get the Pt and Eta arrays of the passing and all probes.
        WARNING: Not recommended to be used for large datasets as the arrays can be very large.

        Parameters
        ----------
            leg : str, optional
                Which leg to get the arrays for. Can be "first", "second", or "both".
                The default is "both".
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
            array_dict: dict
            A dictionary with keys "leg1" and/or "leg2" depending on the leg parameter.
            The values of the dictionary will be tuples that contain the following values:
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
        kwargs_leg1 = {
            "perform_tnp": self._perform_tnp,
            "pt": self.pt1,
            "avoid_ecal_transition_tags": self.avoid_ecal_transition_tags,
            "avoid_ecal_transition_probes": self.avoid_ecal_transition_probes,
            "goldenjson": self.goldenjson,
            "extra_filter": self._extra_filter,
            "extra_filter_args": self._extra_filter_args,
        }
        kwargs_leg2 = {
            "perform_tnp": self._perform_tnp,
            "pt": self.pt2,
            "avoid_ecal_transition_tags": self.avoid_ecal_transition_tags,
            "avoid_ecal_transition_probes": self.avoid_ecal_transition_probes,
            "goldenjson": self.goldenjson,
            "extra_filter": self._extra_filter,
            "extra_filter_args": self._extra_filter_args,
        }

        if leg == "first":
            if compute:
                arrays = _get_and_compute_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg1,
                )
            else:
                arrays = _get_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    **kwargs_leg1,
                )
            return {"leg1": arrays}

        elif leg == "second":
            if compute:
                arrays = _get_and_compute_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg2,
                )
            else:
                arrays = _get_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    **kwargs_leg2,
                )
            return {"leg2": arrays}

        elif leg == "both":
            if compute:
                arrays_leg1 = _get_and_compute_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg1,
                )
                arrays_leg2 = _get_and_compute_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg2,
                )
            else:
                arrays_leg1 = _get_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    **kwargs_leg1,
                )
                arrays_leg2 = _get_arrays_on_leg(
                    events=self.events,
                    perform_tnp=self._perform_tnp,
                    **kwargs_leg2,
                )
            return {"leg1": arrays_leg1, "leg2": arrays_leg2}

        else:
            raise ValueError(
                f"leg must be either 'first', 'second', or 'both'. Got {leg}."
            )

    def get_tnp_histograms(
        self,
        leg="both",
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
            leg : str, optional
                Which leg to get the histograms for. Can be "first", "second", or "both".
                The default is "both".
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
            histogram_dict: dict
                A dictionary with keys "leg1" and/or "leg2" depending on the leg parameter.
                The values of the dictionary will be dictionaries of the form:
                    `{"name": [hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all], ...}`
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
            eta_regions_eta = {"entire": [0, 2.5]}
        if eta_regions_phi is None:
            eta_regions_phi = {"entire": [0, 2.5]}

        kwargs_leg1 = {
            "plateau_cut": plateau_cut,
            "eta_regions_pt": eta_regions_pt,
            "eta_regions_eta": eta_regions_eta,
            "eta_regions_phi": eta_regions_phi,
            "bins": self._bins,
            "perform_tnp": self._perform_tnp,
            "pt": self.pt1,
            "avoid_ecal_transition_tags": self.avoid_ecal_transition_tags,
            "avoid_ecal_transition_probes": self.avoid_ecal_transition_probes,
            "goldenjson": self.goldenjson,
            "extra_filter": self._extra_filter,
            "extra_filter_args": self._extra_filter_args,
        }
        kwargs_leg2 = {
            "plateau_cut": plateau_cut,
            "eta_regions_pt": eta_regions_pt,
            "eta_regions_eta": eta_regions_eta,
            "eta_regions_phi": eta_regions_phi,
            "bins": self._bins,
            "perform_tnp": self._perform_tnp,
            "pt": self.pt2,
            "avoid_ecal_transition_tags": self.avoid_ecal_transition_tags,
            "avoid_ecal_transition_probes": self.avoid_ecal_transition_probes,
            "goldenjson": self.goldenjson,
            "extra_filter": self._extra_filter,
            "extra_filter_args": self._extra_filter_args,
        }

        if leg == "first":
            if compute:
                histograms = _get_and_compute_tnp_histograms_on_leg(
                    events=self.events,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg1,
                )
            else:
                histograms = _get_tnp_histograms_on_leg(
                    events=self.events,
                    **kwargs_leg1,
                )
            return {"leg1": histograms}

        elif leg == "second":
            if compute:
                histograms = _get_and_compute_tnp_histograms_on_leg(
                    events=self.events,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg2,
                )
            else:
                histograms = _get_tnp_histograms_on_leg(
                    events=self.events,
                    **kwargs_leg2,
                )
            return {"leg2": histograms}

        elif leg == "both":
            if compute:
                histograms_leg1 = _get_and_compute_tnp_histograms_on_leg(
                    events=self.events,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg1,
                )
                histograms_leg2 = _get_and_compute_tnp_histograms_on_leg(
                    events=self.events,
                    scheduler=scheduler,
                    progress=progress,
                    **kwargs_leg2,
                )
            else:
                histograms_leg1 = _get_tnp_histograms_on_leg(
                    events=self.events,
                    **kwargs_leg1,
                )
                histograms_leg2 = _get_tnp_histograms_on_leg(
                    events=self.events,
                    **kwargs_leg2,
                )
            return {"leg1": histograms_leg1, "leg2": histograms_leg2}

        else:
            raise ValueError(
                f"leg must be either 'first', 'second', or 'both'. Got {leg}."
            )
