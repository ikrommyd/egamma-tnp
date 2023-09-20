import os

import dask_awkward as dak

from egamma_tnp.utils.dataset import get_nanoevents_file


def get_arrays(events, perform_tnp, *args, **kwargs):
    p1, a1, p2, a2 = perform_tnp(events, *args, **kwargs)

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


def get_and_compute_arrays(events, perform_tnp, scheduler, progress, *args, **kwargs):
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
    ) = get_arrays(events, perform_tnp, *args, **kwargs)

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


def get_tnp_histograms(events, perform_tnp, *args, **kwargs):
    import json
    import os

    import hist
    from hist.dask import Hist

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    ptbins = config["ptbins"]
    etabins = config["etabins"]
    phibins = config["phibins"]

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
    ) = get_arrays(events, perform_tnp, *args, **kwargs)

    ptaxis = hist.axis.Variable(ptbins, name="pt")
    hpt_all = Hist(ptaxis)
    hpt_pass = Hist(ptaxis)

    etaaxis = hist.axis.Variable(etabins, name="eta")
    heta_all = Hist(etaaxis)
    heta_pass = Hist(etaaxis)

    phiaxis = hist.axis.Variable(phibins, name="phi")
    hphi_all = Hist(phiaxis)
    hphi_pass = Hist(phiaxis)

    hpt_pass.fill(pt_pass1)
    hpt_pass.fill(pt_pass2)
    hpt_all.fill(pt_all1)
    hpt_all.fill(pt_all2)
    heta_pass.fill(eta_pass1)
    heta_pass.fill(eta_pass2)
    heta_all.fill(eta_all1)
    heta_all.fill(eta_all2)
    hphi_pass.fill(phi_pass1)
    hphi_pass.fill(phi_pass2)
    hphi_all.fill(phi_all1)
    hphi_all.fill(phi_all2)

    return hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all


def get_and_compute_tnp_histograms(
    events, perform_tnp, scheduler, progress, *args, **kwargs
):
    import dask
    from dask.diagnostics import ProgressBar

    (
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
        hphi_pass,
        hphi_all,
    ) = get_tnp_histograms(events, perform_tnp, *args, **kwargs)

    if progress:
        pbar = ProgressBar()
        pbar.register()

    res = dask.compute(
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
        hphi_pass,
        hphi_all,
        scheduler=scheduler,
    )

    if progress:
        pbar.unregister()

    return res


class BaseTrigger:
    """BaseTrigger class for HLT Trigger efficiency from NanoAOD.

    This class holds the basic methods for all the Tag and Probe classes for different triggers.
    """

    def __init__(
        self,
        names,
        perform_tnp,
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
        if goldenjson is not None and not os.path.isfile(goldenjson):
            raise FileNotFoundError(f"Golden JSON {goldenjson} does not exist.")

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

    def load_events(self, from_root_args={}):
        """Load the events from the names.

        Parameters
        ----------
            from_root_args : dict, optional
                Extra arguments to pass to coffea.nanoevents.NanoEventsFactory.from_root().
                The default is {}.
        """
        from coffea.nanoevents import NanoEventsFactory

        self.events = NanoEventsFactory.from_root(
            self.file,
            permit_dask=True,
            **from_root_args,
        ).events()

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
            pt_pass1: numpy.ndarray or dask_awkward.Array
                The Pt array of the passing probes when the firsts electrons are the tags.
            pt_pass2: numpy.ndarray or dask_awkward.Array
                The Pt array of the passing probes when the seconds electrons are the tags.
            pt_all1: numpy.ndarray or dask_awkward.Array
                The Pt array of all probes when the firsts electrons are the tags.
            pt_all2: numpy.ndarray or dask_awkward.Array
                The Pt array of all probes when the seconds electrons are the tags.
            eta_pass1: numpy.ndarray or dask_awkward.Array
                The Eta array of the passing probes when the firsts electrons are the tags.
            eta_pass2: numpy.ndarray or dask_awkward.Array
                The Eta array of the passing probes when the seconds electrons are the tags.
            eta_all1: numpy.ndarray or dask_awkward.Array
                The Eta array of all probes when the firsts electrons are the tags.
            eta_all2: numpy.ndarray or dask_awkward.Array
                The Eta array of all probes when the seconds electrons are the tags.
            phi_pass1: numpy.ndarray or dask_awkward.Array
                The Phi array of the passing probes when the firsts electrons are the tags.
            phi_pass2: numpy.ndarray or dask_awkward.Array
                The Phi array of the passing probes when the seconds electrons are the tags.
            phi_all1: numpy.ndarray or dask_awkward.Array
                The Phi array of all probes when the firsts electrons are the tags.
            phi_all2: numpy.ndarray or dask_awkward.Array
                The Phi array of all probes when the seconds electrons are the tags.
        """
        if compute:
            return get_and_compute_arrays(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                goldenjson=self.goldenjson,
                scheduler=scheduler,
                progress=progress,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
        else:
            return get_arrays(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                goldenjson=self.goldenjson,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )

    def get_tnp_histograms(self, compute=False, scheduler=None, progress=True):
        """Get the Pt and Eta histograms of the passing and all probes.

        Parameters
        ----------
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
            hpt_pass: hist.Hist or hist.dask.Hist
                The Pt histogram of the passing probes.
            hpt_all: hist.Hist or hist.dask.Hist
                The Pt histogram of all probes.
            heta_pass: hist.Hist or hist.dask.Hist
                The Eta histogram of the passing probes.
            heta_all: hist.Hist or hist.dask.Hist
                The Eta histogram of all probes.
            hphi_pass: hist.Hist or hist.dask.Hist
                The Phi histogram of the passing probes.
            hphi_all: hist.Hist or hist.dask.Hist
                The Phi histogram of all probes.
        """
        if compute:
            return get_and_compute_tnp_histograms(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                goldenjson=self.goldenjson,
                scheduler=scheduler,
                progress=progress,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
        else:
            return get_tnp_histograms(
                events=self.events,
                perform_tnp=self._perform_tnp,
                pt=self.pt,
                goldenjson=self.goldenjson,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
