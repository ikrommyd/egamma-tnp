from __future__ import annotations

import warnings

import awkward as ak
import dask_awkward as dak
import numpy as np
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoAODSchema

from egamma_tnp._base_tagnprobe import BaseSaSNtuples
from egamma_tnp.utils import calculate_photon_SC_eta, custom_delta_r
from egamma_tnp.utils.pileup import create_correction, get_pileup_weight, load_correction


class ScaleAndSmearingNtupleFromNanoAOD(BaseSaSNtuples):
    def __init__(
        self,
        fileset,
        *,
        lead_pt_cut=20,
        sublead_pt_cut=10,
        eta_cut=2.5,
        trigger_paths=None,
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args=None,
        avoid_ecal_transition=False,
    ):
        """Scale and Smearing ntuples from NanoAOD and EGamma NanoAOD.

        Parameters
        ----------
        fileset: dict
            The fileset to calculate the trigger efficiencies for.
        filters: dict
            The names of the filters to calculate the efficiencies for.
        is_photon_filter: dict or None, optional
            Whether the filters to calculate the efficiencies are photon filters. The default is all False.
        trigger_pt: dict or None, optional
            The Pt threshold of the trigger to calculate the efficiencies over that threshold. Required for trigger efficiencies.
            The default is None.
        tags_pt_cut: int or float, optional
            The Pt cut to apply to the tag electrons. The default is 35.
        probes_pt_cuts: int or float, optional
            The Pt threshold of the probe electron to calculate efficiencies over that threshold. The default is None.
        tags_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the tag electrons. The default is 2.5.
        sub_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the probe electrons. The default is 2.5.
        sub_abseta_cut: int or float, optional
            The absolute Eta cut to apply to the probe electrons. The default is 2.5.
        filterbit: dict or None, optional
            The filterbit used to match probes with trigger objects. Required for trigger efficiencies.
            The default is None.
        cutbased_id: str, optional
            ID expression to apply to the probes. An example is "cutBased >= 2".
            If None, no cutbased ID is applied. The default is None.
        extra_zcands_mask: str, optional
            An extra mask to apply to the Z candidates. The default is None.
            Must be of the form `zcands.tag/probe.<mask> & zcands.tag/probe.<mask> & ...`.
        extra_filter: Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args: dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
        use_sc_eta: bool, optional
            Use the supercluster Eta instead of the Eta from the primary vertex. The default is False.
        use_sc_phi: bool, optional
            Use the supercluster Phi instead of the Phi from the primary vertex. The default is False.
        avoid_ecal_transition_tags: bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes: bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        require_event_to_pass_hlt_filter: bool, optional
            Also require the event to have passed the filter HLT filter under study to consider a probe belonging to that event as passing.
            The default is True.
        """

        super().__init__(
            fileset=fileset,
            lead_pt_cut=lead_pt_cut,
            sublead_pt_cut=sublead_pt_cut,
            eta_cut=eta_cut,
            trigger_paths=trigger_paths,
            extra_zcands_mask=extra_zcands_mask,
            extra_filter=extra_filter,
            extra_filter_args=extra_filter_args,
            avoid_ecal_transition=avoid_ecal_transition,
            schemaclass=NanoAODSchema,
        )

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ScaleAndSmearingNtupleFromNanoAOD(Number of files: {n_of_files})"

    def find_lepton_pairs(self, events, mass_range=(50, 130), vars=None):
        if events.metadata.get("isMC") is None:
            events.metadata["isMC"] = hasattr(events, "GenPart")
            if events.metadata["isMC"]:
                events.metadata["sum_genw_presel"] = str(dak.sum(events.genWeight))
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)
        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            lumimask = LumiMask(events.metadata["goldenJSON"])
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        # apply the trigger path filter if specified
        good_events = apply_trigger_paths(events, self.trigger_paths)

        # add superclusterEta to the Photon and Electron objects if not already present
        if "superclusterEta" not in good_events.Photon.fields:
            good_events["Photon", "superclusterEta"] = calculate_photon_SC_eta(good_events.Photon, good_events.PV)
        if "superclusterEta" not in good_events.Electron.fields:
            good_events["Electron", "superclusterEta"] = good_events.Electron.eta + good_events.Electron.deltaEtaSC

        # selecting electrons with a photon matching and passing the pt and eta cuts
        good_events["Electron"] = good_events.Electron[
            (good_events.Electron.photonIdx > -1)
            & (good_events.Electron.pt > self.sublead_pt_cut)
            & (np.abs(good_events.Electron.superclusterEta) < self.eta_cut)
        ]

        # avoid the ECAL transition region for the electrons with an eta cut
        if self.avoid_ecal_transition:
            good_events["Electron"] = good_events.Electron[
                ~((np.abs(good_events.Electron.superclusterEta) > 1.4442) & (np.abs(good_events.Electron.superclusterEta) < 1.566))
            ]

        good_events = good_events[ak.num(good_events.Electron) >= 2]
        electrons = good_events.Electron
        electron_fields = list(electrons.fields)

        # get the matched photons
        matched_photons = good_events.Photon[electrons.photonIdx]
        photon_fields = list(matched_photons.fields)

        # update the vars dictionary with the fields to save
        if vars is None:
            vars = {"Electron": electron_fields, "Photon": photon_fields}
        else:
            if "Electron" not in vars or vars["Electron"] == "all":
                vars["Electron"] = electron_fields
            if "Photon" not in vars:
                vars["Photon"] = []
                warnings.warn("vars does not contain 'Photon' key, not saving photon variables", UserWarning)
            elif vars["Photon"] == "all":
                vars["Photon"] = photon_fields

        # add GenPart information if available
        if good_events.metadata.get("isMC"):
            electrons["gen_pt"] = good_events.GenPart[electrons.genPartIdx].pt
            vars["Electron"] += ["gen_pt"]

        # add the matched photon variables to the electrons
        for var in vars["Photon"]:
            electrons[f"pho_{var}"] = matched_photons[var]

        good_events["Electron"] = electrons
        sorted_electrons = good_events.Electron[ak.argsort(good_events.Electron.pt, ascending=False)]

        dielectrons = process_zcands(leptons=sorted_electrons, mass_range=mass_range, lead_pt_cut=self.lead_pt_cut, prefixes=("ele_lead", "ele_sublead"))
        dielectrons = save_event_variables(good_events, dielectrons, vars=vars)
        dielectrons = apply_pileup_weights(dielectrons, good_events)

        # flatten the output
        output = {}
        for field in ak.fields(dielectrons):
            prefix = {"ele_lead": "lead", "ele_sublead": "sublead"}.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(dielectrons[field]):
                    if subfield.startswith("pho_") and (subfield[4:] not in vars.get("Photon", [])):
                        continue
                    elif not (subfield.startswith("pho_")) and subfield not in vars.get("Electron", []):
                        continue
                    else:
                        output[f"{prefix}_{subfield}"] = dielectrons[field][subfield]
            else:
                output[field] = dielectrons[field]

        return dak.zip(output)

    @staticmethod
    def _trigger_match(leptons, trigobjs, pdgid, pt, filterbit):
        pass_pt = trigobjs.pt > pt
        pass_id = abs(trigobjs.id) == pdgid
        pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
        trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
        delta_r = leptons.metric_table(trigger_cands, metric=custom_delta_r)
        pass_delta_r = delta_r < 0.1
        trig_matched_locs = dak.any(pass_delta_r, axis=2)

        return trig_matched_locs


def process_zcands(leptons, lead_pt_cut=20, mass_range=(50, 130), prefixes=("ele_lead", "ele_sublead")):
    dileptons = ak.combinations(leptons, 2, fields=[prefixes[0], prefixes[1]])
    # Apply the cut on the leading leptons's pT
    dileptons = dileptons[dileptons[prefixes[0]].pt > lead_pt_cut]

    # Combine four-momenta of the two leptons
    dilepton_4mom = dileptons[prefixes[0]] + dileptons[prefixes[1]]
    dileptons["pt"] = dilepton_4mom.pt
    dileptons["eta"] = dilepton_4mom.eta
    dileptons["phi"] = dilepton_4mom.phi
    dileptons["mass"] = dilepton_4mom.mass
    dileptons["charge"] = dilepton_4mom.charge

    # Calculate rapidity
    dilepton_pz = dilepton_4mom.z
    dilepton_e = dilepton_4mom.energy
    dileptons["rapidity"] = 0.5 * np.log((dilepton_e + dilepton_pz) / (dilepton_e - dilepton_pz))

    # Sort dielectron candidates by pT in descending order
    dileptons = dileptons[ak.argsort(dileptons.pt, ascending=False)]

    dileptons = dileptons[(dileptons.mass > mass_range[0]) & (dileptons.mass < mass_range[1])]
    dileptons = dileptons[dileptons.ele_lead.charge != dileptons.ele_sublead.charge]
    selection_mask = ~ak.is_none(dileptons)
    dileptons = dileptons[selection_mask]

    return ak.firsts(dileptons)


def save_event_variables(events, dileptons, vars=None):
    dileptons["event"] = events.event
    dileptons["lumi"] = events.luminosityBlock
    dileptons["run"] = events.run
    # nPV just for validation of pileup reweighting
    dileptons["nPV"] = events.PV.npvs
    if hasattr(events.Rho, "fixedGridRhoAll"):
        dileptons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
    # annotate dielectrons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
    if events.metadata.get("isMC"):
        dileptons["genWeight"] = events.genWeight
        dileptons["nTrueInt"] = events.Pileup.nTrueInt
        dileptons["dZ"] = events.GenVtx.z - events.PV.z
    # Fill zeros for data because there is no GenVtx for data, obviously
    else:
        dileptons["dZ"] = ak.zeros_like(events.PV.z)

    # save variables from other collections if specified
    for collection in vars.keys() if vars is not None else []:
        if collection not in ["Electron", "Photon", "Muon"]:
            for var in events[collection].fields if vars[collection] == "all" else vars[collection]:
                dileptons[f"{collection}_{var}"] = events[collection][var]

    return dileptons


def apply_pileup_weights(dileptons, events):
    if events.metadata.get("isMC"):
        if "pileupJSON" in events.metadata:
            print(f"Loading pileup correction from {events.metadata['pileupJSON']}")
            pileup_corr = load_correction(events.metadata["pileupJSON"])
        elif "pileupData" in events.metadata and "pileupMC" in events.metadata:
            pileup_corr = create_correction(events.metadata["pileupData"], events.metadata["pileupMC"])
        else:
            pileup_corr = None
        if pileup_corr is not None:
            pileup_weight_nom, pileup_weight_up, pileup_weight_down = get_pileup_weight(dileptons.nTrueInt, pileup_corr, syst=True)
            dileptons["weight_central"] = pileup_weight_nom
            dileptons["weight_central_PileupUp"] = pileup_weight_up
            dileptons["weight_central_PileupDown"] = pileup_weight_down

            dileptons["weight"] = pileup_weight_nom * dileptons["genWeight"]
            dileptons["weight_PileupUp"] = pileup_weight_up * dileptons["genWeight"]
            dileptons["weight_PileupDown"] = pileup_weight_down * dileptons["genWeight"]

        else:
            dileptons["weight_central"] = ak.ones_like(dileptons.pt)
            dileptons["weight"] = dileptons["genWeight"]

    return dileptons


def apply_trigger_paths(events, trigger_paths):
    if trigger_paths is not None:
        trigger_names = []
        if isinstance(trigger_paths, str):
            trigger_paths = [trigger_paths]
        # Remove wildcards from trigger paths and find the corresponding fields in events.HLT
        for trigger in trigger_paths:
            actual_trigger = trigger.replace("*", "")
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
