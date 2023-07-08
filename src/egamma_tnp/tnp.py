from ._tnputils import get_and_compute_tnp_histograms, get_tnp_histograms
from .utils import get_events


class TagNProbe:
    def __init__(self, dataset, *, local=False, goldenjson=None):
        """Create a TagNProbe object.

        Parameters
        ----------
        dataset : str or list of str
            Dataset name(s) to be used. If `local` is False, this should be a DAS query.
            If `local` is True, this should be a list of file paths.
        local : bool, optional
            Whether to use local files or DAS. The default is False.
        goldenjson : str, optional
            Path to the golden JSON file for luminosity masking. The default is None.
        """
        self.events, fnames = get_events(dataset, local)
        self.files = list(fnames.keys())
        self.goldenjson = goldenjson

    def __repr__(self):
        return f"TagNProbe(files={self.files}, goldenjson={self.goldenjson})"

    def get_tnp_histograms(self, compute=False, scheduler="threads", progress=True):
        """Get the Pt and Eta histograms of the passing and all probes.

        Parameters
        ----------
        compute : bool, optional
            Whether to return the computed histograms or the dask graphs. The default is False.
        scheduler : str, optional
            The dask scheduler to use. The default is "threads".
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
