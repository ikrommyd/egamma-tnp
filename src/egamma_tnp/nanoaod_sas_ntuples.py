from __future__ import annotations

import warnings

import dask_awkward as dak
import numpy as np
from coffea.nanoevents import NanoAODSchema

from egamma_tnp._base_ntuplizer import BaseNTuplizer
from egamma_tnp.utils import calculate_photon_SC_eta
from egamma_tnp.utils.pileup import apply_pileup_weights


class ScaleAndSmearingNTuplesFromNanoAOD(BaseNTuplizer):
    def __init__(
        self,
        fileset,
        *,
        lead_pt_cut=20,
        sublead_pt_cut=10,
        eta_cut=2.5,
        trigger_paths=None,
        extra_filter=None,
        extra_filter_args=None,
        avoid_ecal_transition=False,
    ):
        """Scale and smear NTuples from NanoAOD.

        Parameters
        ----------
            fileset: dict
                Dictionary specifying the input files to process.
            lead_pt_cut: float, optional
                Minimum transverse momentem for the leading electron. The default is 20.
            sublead_pt_cut: float, optional
                Minimum transverse momentum for the subleading electron. The default is 10.
            eta_cut: float, optional
                Maximum absolute pseudorapidity for electrons. The default is 2.5.
            trigger_paths: list or str, optional
                List of trigger path names to apply as event selection. The default is None.
            extra_filter: Callable, optional
                Function to further filter events. The default is None.
                Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
            extra_filter_args: dict, optional
                Arguments to pass to extra_filter. The default is {}.
            avoid_ecal_transition: bool, optional
                Whether to exclude electrons in the ECAL transition region. The default is False.
        """

        super().__init__(
            fileset=fileset,
            schemaclass=NanoAODSchema,
        )

        self.lead_pt_cut = lead_pt_cut
        self.sublead_pt_cut = sublead_pt_cut
        self.eta_cut = eta_cut
        self.trigger_paths = trigger_paths
        self.extra_filter = extra_filter
        if extra_filter_args is None:
            extra_filter_args = {}
        self.extra_filter_args = extra_filter_args
        self.avoid_ecal_transition = avoid_ecal_transition

    def __repr__(self):
        n_of_files = 0
        for dataset in self.fileset.values():
            n_of_files += len(dataset["files"])
        return f"ScaleAndSmearingNTuplesFromNanoAOD(Number of files: {n_of_files})"

    def make_ntuples(self, events, mass_range, vars):
        if events.metadata.get("isMC") is None:
            events.metadata["isMC"] = hasattr(events, "GenPart")
            if events.metadata.get("isMC") and "genWeight" in events.fields:
                sum_genw_before_presel = dak.sum(events.genWeight)
            else:
                sum_genw_before_presel = 1.0
        if self.extra_filter is not None:
            events = self.extra_filter(events, **self.extra_filter_args)

        if events.metadata.get("goldenJSON") and not events.metadata.get("isMC"):
            events = self.apply_goldenJSON(events)

        # apply the trigger path filter if specified
        good_events = self.apply_trigger_paths(events, self.trigger_paths)

        # add superclusterEta to the Photon and Electron objects if not already present
        if "superclusterEta" not in good_events.Photon.fields:
            good_events["Photon", "superclusterEta"] = calculate_photon_SC_eta(good_events.Photon, good_events.PV)
        if "superclusterEta" not in good_events.Electron.fields:
            good_events["Electron", "superclusterEta"] = good_events.Electron.eta + good_events.Electron.deltaEtaSC

        # selecting electrons with a photon matching and passing the pt and eta cuts
        good_events["Electron"] = good_events.Electron[
            (good_events.Electron.photonIdx != -1)
            & (good_events.Electron.pt > self.sublead_pt_cut)
            & (np.abs(good_events.Electron.superclusterEta) < self.eta_cut)
        ]

        # avoid the ECAL transition region for the electrons with an eta cut
        if self.avoid_ecal_transition:
            good_events["Electron"] = good_events.Electron[
                ~((np.abs(good_events.Electron.superclusterEta) > 1.4442) & (np.abs(good_events.Electron.superclusterEta) < 1.566))
            ]

        good_events = good_events[dak.num(good_events.Electron) >= 2]
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
                warnings.warn("vars does not contain 'Photon' key, not saving photon variables", UserWarning, stacklevel=2)
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
        sorted_electrons = good_events.Electron[dak.argsort(good_events.Electron.pt, ascending=False)]

        dielectrons = ScaleAndSmearingNTuplesFromNanoAOD._process_leptons(
            leptons=sorted_electrons, mass_range=mass_range, lead_pt_cut=self.lead_pt_cut, prefixes=("ele_lead", "ele_sublead")
        )
        dielectrons = ScaleAndSmearingNTuplesFromNanoAOD._save_event_variables(good_events, dielectrons, vars=vars)
        dielectrons = apply_pileup_weights(dielectrons, good_events, sum_genw_before_presel=sum_genw_before_presel, syst=True)

        # flatten the output
        output = {}
        for field in dak.fields(dielectrons):
            prefix = {"ele_lead": "lead", "ele_sublead": "sublead"}.get(field, "")
            if len(prefix) > 0:
                for subfield in dak.fields(dielectrons[field]):
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
    def _process_leptons(leptons, lead_pt_cut, mass_range, prefixes):
        dileptons = dak.combinations(leptons, 2, fields=[prefixes[0], prefixes[1]])
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
        dilepton_pz = dilepton_4mom.pz
        dilepton_e = dilepton_4mom.energy
        dileptons["rapidity"] = 0.5 * np.log((dilepton_e + dilepton_pz) / (dilepton_e - dilepton_pz))

        # Sort dielectron candidates by pT in descending order
        dileptons = dileptons[dak.argsort(dileptons.pt, ascending=False)]

        dileptons = dileptons[(dileptons.mass > mass_range[0]) & (dileptons.mass < mass_range[1])]
        dileptons = dileptons[dileptons.ele_lead.charge != dileptons.ele_sublead.charge]
        selection_mask = ~dak.is_none(dileptons)
        dileptons = dileptons[selection_mask]

        return dak.firsts(dileptons)

    @staticmethod
    def _save_event_variables(events, dileptons, vars):
        dileptons["event"] = events.event
        dileptons["lumi"] = events.luminosityBlock
        dileptons["run"] = events.run
        # nPV and fixedGridRhoAll just for validation of pileup reweighting
        dileptons["nPV"] = events.PV.npvs
        if hasattr(events, "Rho") and hasattr(events.Rho, "fixedGridRhoAll"):
            dileptons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
        elif hasattr(events, "fixedGridRhoFastjetAll"):  # NanoAODv9
            dileptons["fixedGridRhoFastjetAll"] = events.fixedGridRhoFastjetAll
        # annotate dielectrons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
        if events.metadata.get("isMC"):
            dileptons["genWeight"] = events.genWeight
            dileptons["nTrueInt"] = events.Pileup.nTrueInt
            dileptons["dZ"] = events.GenVtx.z - events.PV.z
        # Fill zeros for data because there is no GenVtx for data, obviously
        else:
            dileptons["dZ"] = dak.zeros_like(events.PV.z)

        # save variables from other collections if specified
        for collection in vars.keys() if vars is not None else []:
            if collection not in ["Electron", "Photon", "Muon"]:
                for var in events[collection].fields if vars[collection] == "all" else vars[collection]:
                    dileptons[f"{collection}_{var}"] = events[collection][var]

        return dileptons
