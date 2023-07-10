from ._tnputils import get_and_compute_tnp_histograms, get_tnp_histograms
from .utils import get_events


class TagNProbe:
    def __init__(
        self,
        names,
        *,
        goldenjson=None,
        toquery=False,
        redirect=False,
        custom_redirector="root://cmsxrootd.fnal.gov/",
        invalid=False,
    ):
        """Tag and Probe for HLT Trigger efficiency from NanoAOD.

        Parameters
        ----------
        names : str or list of str
            The dataset names to query that can contain wildcards or a list of file paths.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        toquery : bool, optional
            Whether to query DAS for the dataset names. The default is False.
        redirect : bool, optional
            Whether to add an xrootd redirector to the files. The default is False.
        custom_redirector : str, optional
            The xrootd redirector to add to the files. The default is "root://cmsxrootd.fnal.gov/".
            Only used if redirect is True.
        invalid : bool, optional
            Whether to include invalid files. The default is False.
            Only used if toquery is True.
        """
        self.names = names
        self.goldenjson = goldenjson
        self.toquery = toquery
        self.redirect = redirect
        self.custom_redirector = custom_redirector
        self.invalid = invalid
        self.events = None
        self.files = None

    def __repr__(self):
        if self.files:
            return f"TagNProbe(Events: {self.events}, Number of files: {self.files}, (Golden JSON: {self.goldenjson})"
        else:
            return f"TagNProbe(Events: {self.events}, Number of files: not loaded, (Golden JSON: {self.goldenjson})"

    def load_events(self):
        """Load the events from the names."""
        self.events, self.files = get_events(
            self.names,
            toquery=self.toquery,
            redirect=self.redirect,
            custom_redirector=self.custom_redirector,
            invalid=self.invalid,
        )

    def get_tnp_histograms(self, compute=False, scheduler=None, progress=True):
        """Get the Pt and Eta histograms of the passing and all probes.

        Parameters
        ----------
        compute : bool, optional
            Whether to return the computed histograms or the dask graphs. The default is False.
        scheduler : str, optional
            The dask scheduler to use. The default is None.
        progress : bool, optional
            Whether to show a progress bar if `compute` is True. The default is True.
        """
        if compute:
            return get_and_compute_tnp_histograms(
                events=self.events,
                goldenjson=self.goldenjson,
                scheduler=scheduler,
                progress=progress,
            )
        else:
            return get_tnp_histograms(events=self.events, goldenjson=self.goldenjson)
