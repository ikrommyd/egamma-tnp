from __future__ import annotations

from functools import partial

from coffea.dataset_tools import apply_to_fileset
from coffea.lumi_tools import LumiMask


class BaseNTuplizer:
    """Base class for any NTuplizer classes."""

    def __init__(
        self,
        fileset,
        schemaclass,
    ):
        self.fileset = fileset
        self.schemaclass = schemaclass

    def make_ntuples(self, events, cut_and_count, mass_range, vars):
        """Make Dielectron NTuples.

        Parameters
        ----------
            events : awkward.Array or dask_awarkward.Array
                events read using coffea.nanoevents.NanoEventsFactory.from_root
            mass_range: uple of two ints or floats, optional
                The allowed mass range of the dielectron pairs.
                It represents the upper and lower bounds of the mass range.
            vars : dict
                A dict of variables to return.
                The keys are the object names and the values are the list of variables to return for each object.

        Returns
        _______
            array : awkward.Array or dask_awarkward.Array
                An array with fields specified in `vars`.
        """
        raise NotImplementedError("make_ntuples method must be implemented.")

    def get_ntuples(
        self,
        mass_range=None,
        vars=None,
        flat=False,
        uproot_options=None,
        compute=False,
        scheduler=None,
        progress=False,
    ):
        """
        Get arrays of lepton pair and event-level variables.

        WARNING: Not recommended for large datasets as the arrays can be very large.

        Parameters
        ----------
            mass_range : tuple of two ints/floats, optional
                Allowed mass range for the dilepton pairs.
                It represents the lower and upper bounds.
                Default is (50, 130).
            vars : dict, optional
                A dict of variables to return.
                The keys are the object names and the values are the list of variables to return for each.
            flat : bool, optional
                Whether to return flat arrays. Default is False.
            uproot_options : dict, optional
                Options to pass to uproot. For file access reports, use {"allow_read_errors_with_report": True}.
            compute : bool, optional
                Whether to return computed arrays or delayed arrays. Default is False.
            scheduler : str, optional
                Dask scheduler to use if compute is True. Default is None.
            progress : bool, optional
                Show a progress bar if compute is True. Default is False.

        Returns
        -------
            A tuple of the form (arrays, report) if `allow_read_errors_with_report` is True, otherwise just arrays.
            arrays: a zip object containing the fields specified in `vars` and a field with a boolean array for each filter.
                It will also contain the `pair_mass` field if cut_and_count is False.
            report: dict of awkward arrays of the same form as fileset.
                For each dataset an awkward array that contains information about the file access is present.
        """
        if uproot_options is None:
            uproot_options = {}
        if mass_range is None:
            mass_range = (50, 130)

        if flat:
            from egamma_tnp.utils.histogramming import flatten_array

            def data_manipulation(events):
                return flatten_array(self.make_ntuples(events, mass_range=mass_range, vars=vars))
        else:
            data_manipulation = partial(self.make_ntuples, mass_range=mass_range, vars=vars)

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

    def apply_trigger_paths(self, events, trigger_paths):
        if trigger_paths is not None:
            trigger_names = []
            if isinstance(trigger_paths, str):
                trigger_paths = [trigger_paths]
            # Remove wildcards from trigger paths and find the corresponding fields in events.HLT
            for trigger in trigger_paths:
                actual_trigger = trigger.replace("*", "").replace("HLT_", "")
                for field in events.HLT.fields:
                    if field.startswith(actual_trigger):
                        trigger_names.append(field)
            # Select events that pass any of the specified trigger paths
            trigger_mask = events.run < 0
            for trigger in trigger_names:
                trigger_mask = trigger_mask | getattr(events.HLT, trigger)

            good_events = events[trigger_mask]
        else:
            good_events = events

        return good_events

    def apply_goldenJSON(self, events):
        lumimask = LumiMask(events.metadata["goldenJSON"])
        mask = lumimask(events.run, events.luminosityBlock)
        events = events[mask]
        return events
