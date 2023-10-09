import dask
import uproot
from distributed import Client

import egamma_tnp
from egamma_tnp.triggers import ElePt_CaloIdVT_GsfTrkIdT


def l1_filter(events, l1_seeds):
    if isinstance(l1_seeds, str):
        l1_seeds = [l1_seeds]
    for seed in l1_seeds:
        mask = events.L1[seed]
        events = events[mask]
    return events


egamma_tnp.config.set(
    "ptbins",
    [
        5,
        10,
        15,
        20,
        22,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        45,
        50,
        60,
        80,
        100,
        105,
        110,
        115,
        120,
        125,
        130,
        135,
        140,
        145,
        150,
        250,
        400,
    ],
)

if __name__ == "__main__":
    client = Client()

    # from lpcjobqueue import LPCCondorCluster

    # cluster = LPCCondorCluster(ship_env=True)
    # cluster.adapt(minimum=1, maximum=100)
    # client = Client(cluster)

    tag_n_probe = ElePt_CaloIdVT_GsfTrkIdT(
        [
            "/EGamma0/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
            "/EGamma1/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
            "/EGamma0/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
            "/EGamma1/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
            "/EGamma0/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
            "/EGamma1/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
            "/EGamma0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
            "/EGamma1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
            "/EGamma0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
            "/EGamma1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
            "/EGamma0/Run2023D-PromptReco-v1/NANOAOD",
            "/EGamma1/Run2023D-PromptReco-v1/NANOAOD",
            "/EGamma0/Run2023D-PromptReco-v2/NANOAOD",
            "/EGamma1/Run2023D-PromptReco-v2/NANOAOD",
        ],
        115,
        goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
        toquery=True,
        redirect=False,
        preprocess=True,
        preprocess_args={"maybe_step_size": 100_000},
    )

    print("Starting to load events")
    tag_n_probe.load_events(from_root_args={"uproot_options": {"timeout": 120}})
    print(tag_n_probe)

    res = tag_n_probe.get_tnp_histograms()
    hpt_pass, hpt_all, heta_pass, heta_all, hphi_pass, hphi_all = dask.compute(*res)

    print(f"Passing probes: {hpt_pass.sum(flow=True)}")
    print(f"All probes: {hpt_all.sum(flow=True)}")

    with uproot.recreate("root_files/Run2023.root") as file:
        file["hpt_pass"] = hpt_pass
        file["hpt_all"] = hpt_all
        file["heta_pass"] = heta_pass
        file["heta_all"] = heta_all
        file["hphi_pass"] = hphi_pass
        file["hphi_all"] = hphi_all
