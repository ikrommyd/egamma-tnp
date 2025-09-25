from __future__ import annotations

import awkward as ak
import numpy as np

from egamma_tnp.utils import calculate_photon_SC_eta


def sas_ntuples(events):
    good_events = events[events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ]
    good_events["Electron", "superclusterEta"] = good_events.Electron.eta + good_events.Electron.deltaEtaSC
    good_events["Photon", "superclusterEta"] = calculate_photon_SC_eta(good_events.Photon, good_events.PV)

    # selecting electrons with a photon matching and passing the pt and eta cuts
    good_events["Electron"] = good_events.Electron[
        (good_events.Electron.photonIdx > -1) & (good_events.Electron.pt > 12.0) & (np.abs(good_events.Electron.superclusterEta) < 2.17)
    ]

    # avoid the ECAL transition region for the electrons with an eta cut
    good_events["Electron"] = good_events.Electron[
        ~((np.abs(good_events.Electron.superclusterEta) > 1.4442) & (np.abs(good_events.Electron.superclusterEta) < 1.566))
    ]

    good_events = good_events[ak.num(good_events.Electron) >= 2]
    electrons = good_events.Electron
    matched_photons = good_events.Photon[electrons.photonIdx]

    vars = {
        "Photon": [
            "pt",
            "eta",
            "phi",
            "seedGain",
            "superclusterEta",
            "r9",
            "sieie",
            "hoe",
            "hoe_PUcorr",
            "s4",
            "phiWidth",
            "etaWidth",
            "sieip",
            "sipip",
            "esEffSigmaRR",
            "esEnergyOverRawE",
            "cutBased",
            "mvaID",
            "mvaID_WP80",
            "mvaID_WP90",
        ],
        "Electron": [
            "pt",
            "eta",
            "phi",
            "seedGain",
            "charge",
            "superclusterEta",
            "r9",
            "sieie",
            "hoe",
            "convVeto",
            "eInvMinusPInv",
            "fbrem",
            "cutBased",
            "cutBased_HEEP",
            "mvaIso",
            "mvaIso_WP80",
            "mvaIso_WP90",
            "mvaIso_WPHZZ",
            "mvaNoIso",
            "mvaNoIso_WP80",
            "mvaNoIso_WP90",
            "pfRelIso03_all",
            "pfRelIso03_chg",
            "dr03EcalRecHitSumEt",
            "dr03HcalDepth1TowerSumEt",
            "dr03TkSumPt",
        ],
        "PuppiMET": ["pt", "phi", "sumEt"],
        "Rho": "all",
    }

    for var in vars["Photon"]:
        electrons[f"pho_{var}"] = matched_photons[var]

    good_events["Electron"] = electrons
    sorted_electrons = good_events.Electron[ak.argsort(good_events.Electron.pt, ascending=False)]

    prefixes = ("ele_lead", "ele_sublead")
    mass_range = (60, 120)
    dileptons = ak.combinations(sorted_electrons, 2, fields=[prefixes[0], prefixes[1]])
    # Apply the cut on the leading leptons's pT
    dileptons = dileptons[dileptons[prefixes[0]].pt > 23.0]

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
    dielectrons = ak.firsts(dileptons)

    dielectrons["event"] = good_events.event
    dielectrons["lumi"] = good_events.luminosityBlock
    dielectrons["run"] = good_events.run
    # nPV and fixedGridRhoAll just for validation of pileup reweighting
    dielectrons["nPV"] = good_events.PV.npvs
    if hasattr(good_events.Rho, "fixedGridRhoAll"):
        dielectrons["fixedGridRhoAll"] = good_events.Rho.fixedGridRhoAll
    # annotate dielectrons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
    dielectrons["dZ"] = ak.zeros_like(good_events.PV.z)

    # save variables from other collections if specified
    for collection in vars.keys() if vars is not None else []:
        if collection not in ["Electron", "Photon", "Muon"]:
            for var in good_events[collection].fields if vars[collection] == "all" else vars[collection]:
                dielectrons[f"{collection}_{var}"] = good_events[collection][var]

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

    return ak.zip(output)
