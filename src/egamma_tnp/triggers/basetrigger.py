import json
import os

from egamma_tnp.utils.dataset import get_nanoevents_file


class BaseTrigger:
    """BaseTrigger class for HLT Trigger efficiency from NanoAOD.

    This class holds the basic methods for all the Tag and Probe classes for different single and double electron triggers.
    """

    def __init__(
        self,
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
    ):
        self.names = names
        self._perform_tnp = perform_tnp
        self.avoid_ecal_transition_tags = avoid_ecal_transition_tags
        self.avoid_ecal_transition_probes = avoid_ecal_transition_probes
        self.goldenjson = goldenjson
        self.events = None
        self._toquery = toquery
        self._redirector = redirector
        self._preprocess = preprocess
        self._preprocess_args = preprocess_args
        self._extra_filter = extra_filter
        self._extra_filter_args = extra_filter_args

        self.file = get_nanoevents_file(
            self.names,
            toquery=self._toquery,
            redirector=self._redirector,
            preprocess=self._preprocess,
            preprocess_args=self._preprocess_args,
        )
        self.report = None

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

    def load_events(self, from_root_args=None, allow_read_errors_with_report=False):
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
        if allow_read_errors_with_report:
            if "uproot_options" in from_root_args:
                from_root_args["uproot_options"]["allow_read_errors_with_report"] = True
            else:
                from_root_args["uproot_options"] = {
                    "allow_read_errors_with_report": True
                }
            self.events, self.report = NanoEventsFactory.from_root(
                self.file,
                return_read_report=True,
                **from_root_args,
            ).events()

        self.events = NanoEventsFactory.from_root(
            self.file,
            **from_root_args,
        ).events()
