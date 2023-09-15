import dask_awkward as dak
from coffea.lumi_tools import LumiMask

from egamma_tnp.triggers.basetrigger import BaseTrigger


def apply_lumimasking(events, goldenjson):
    lumimask = LumiMask(goldenjson)
    mask = lumimask(events.run, events.luminosityBlock)
    return events[mask]


def filter_events(events, pt):
    events = events[dak.num(events.Electron) >= 2]
    abs_eta = abs(events.Electron.eta)
    pass_eta_ebeegap = (abs_eta < 1.4442) | (abs_eta > 1.566)
    pass_tight_id = events.Electron.cutBased == 4
    pass_pt = events.Electron.pt > pt
    pass_eta = abs_eta <= 2.5
    pass_selection = pass_pt & pass_eta & pass_eta_ebeegap & pass_tight_id
    n_of_tags = dak.sum(pass_selection, axis=1)
    good_events = events[n_of_tags >= 2]
    good_locations = pass_selection[n_of_tags >= 2]

    return good_events, good_locations


def trigger_match(electrons, trigobjs, pt):
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == 11
    filterbit = 1
    pass_wptight = trigobjs.filterBits & (0x1 << filterbit) == 2**filterbit
    trigger_cands = trigobjs[pass_pt & pass_id & pass_wptight]

    delta_r = electrons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = dak.sum(dak.sum(pass_delta_r, axis=1), axis=1)
    trig_matched_locs = n_of_trigger_matches >= 1

    return trig_matched_locs


def find_probes(tags, probes, trigobjs, pt):
    trig_matched_tag = trigger_match(tags, trigobjs, pt)
    tags = tags[trig_matched_tag]
    probes = probes[trig_matched_tag]
    trigobjs = trigobjs[trig_matched_tag]

    dr = tags.delta_r(probes)
    mass = (tags + probes).mass

    in_mass_window = abs(mass - 91.1876) < 30
    opposite_charge = tags.charge * probes.charge == -1

    isZ = in_mass_window & opposite_charge
    dr_condition = dr > 0.0

    all_probes = probes[isZ & dr_condition]
    trig_matched_probe = trigger_match(all_probes, trigobjs, pt)
    passing_probes = all_probes[trig_matched_probe]

    return passing_probes, all_probes


def perform_tnp(events, pt, goldenjson, extra_filter, extra_filter_args):
    if extra_filter is not None:
        events = extra_filter(events, **extra_filter_args)
    if goldenjson is not None:
        events = apply_lumimasking(events, goldenjson)
    good_events, good_locations = filter_events(events, pt)
    ele_for_tnp = good_events.Electron[good_locations]

    zcands1 = dak.combinations(ele_for_tnp, 2, fields=["tag", "probe"])
    zcands2 = dak.combinations(ele_for_tnp, 2, fields=["probe", "tag"])
    p1, a1 = find_probes(zcands1.tag, zcands1.probe, good_events.TrigObj, pt)
    p2, a2 = find_probes(zcands2.tag, zcands2.probe, good_events.TrigObj, pt)

    return p1, a1, p2, a2


def get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args):
    p1, a1, p2, a2 = perform_tnp(
        events, pt, goldenjson, extra_filter, extra_filter_args
    )

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


def get_and_compute_arrays(
    events, pt, goldenjson, scheduler, progress, extra_filter, extra_filter_args
):
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
    ) = get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args)

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


def get_tnp_histograms(events, pt, goldenjson, extra_filter, extra_filter_args):
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
    ) = get_arrays(events, pt, goldenjson, extra_filter, extra_filter_args)

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
    events, pt, goldenjson, scheduler, progress, extra_filter, extra_filter_args
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
    ) = get_tnp_histograms(events, pt, goldenjson, extra_filter, extra_filter_args)

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


class ElePt_WPTight_Gsf(BaseTrigger):
    def __init__(
        self,
        names,
        trigger_pt,
        *,
        goldenjson=None,
        toquery=False,
        redirect=False,
        custom_redirector="root://cmsxrootd.fnal.gov/",
        invalid=False,
        preprocess=False,
        preprocess_args={},
        extra_filter=None,
        extra_filter_args={},
    ):
        """Tag and Probe efficiency for HLT_ElePt_WPTight_Gsf trigger from NanoAOD.

        Parameters
        ----------
            names : str or list of str
                The dataset names to query that can contain wildcards or a list of file paths.
            trigger_pt : int or float
                The Pt threshold of the trigger.
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
            preprocess : bool, optional
                Whether to preprocess the files using coffea.dataset_tools.preprocess().
                The default is False.
            preprocess_args : dict, optional
                Extra arguments to pass to coffea.dataset_tools.preprocess(). The default is {}.
            extra_filter : Callable, optional
                A extra function to filter the events. The default is None.
                Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
            extra_filter_args : dict, optional
                Extra arguments to pass to extra_filter. The default is {}.
        """
        self.pt = trigger_pt - 1
        super().__init__(
            names=names,
            goldenjson=goldenjson,
            toquery=toquery,
            redirect=redirect,
            custom_redirector=custom_redirector,
            invalid=invalid,
            preprocess=preprocess,
            preprocess_args=preprocess_args,
            extra_filter=extra_filter,
            extra_filter_args=extra_filter_args,
        )

    def __repr__(self):
        if self.events is None:
            return f"HLT_Ele{self.pt + 1}_WPTight_Gsf(Events: not loaded, Number of files: {len(self.file)}, Golden JSON: {self.goldenjson})"
        else:
            return f"HLT_Ele{self.pt + 1}_WPTight_Gsf(Events: {self.events}, Number of files: {len(self.file)}, Golden JSON: {self.goldenjson})"

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
                pt=self.pt,
                goldenjson=self.goldenjson,
                extra_filter=self._extra_filter,
                extra_filter_args=self._extra_filter_args,
            )
