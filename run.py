import uproot
from distributed import Client

import egamma_tnp
from egamma_tnp.triggers import ElePt_WPTight_Gsf

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
        150,
        250,
        400,
    ],
)

samples2022 = {
    "2022C": ["/EGamma/Run2022C-PromptNanoAODv10_v1-v1/NANOAOD"],
    "2022D": [
        "/EGamma/Run2022D-PromptNanoAODv10_v1-v1/NANOAOD",
        "/EGamma/Run2022D-PromptNanoAODv10_v2-v1/NANOAOD",
    ],
    "2022E": ["/EGamma/Run2022E-PromptNanoAODv10_v1-v2/NANOAOD"],
    "2022F": ["/EGamma/Run2022F-PromptNanoAODv10_v1-v2/NANOAOD"],
    "2022G": ["/EGamma/Run2022G-PromptNanoAODv10_v1-v1/NANOAOD"],
}

samples2023 = {
    "2023B": [
        "/EGamma0/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
        "/EGamma1/Run2023B-PromptNanoAODv11p9_v1-v1/NANOAOD",
    ],
    "2023C": [
        "/EGamma0/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
    ],
    "2023D": [
        "/EGamma0/Run2023D-PromptReco-v1/NANOAOD",
        "/EGamma1/Run2023D-PromptReco-v1/NANOAOD",
        "/EGamma0/Run2023D-PromptReco-v2/NANOAOD",
        "/EGamma1/Run2023D-PromptReco-v2/NANOAOD",
    ],
}

pt = 32

if __name__ == "__main__":
    # client = Client()

    from lpcjobqueue import LPCCondorCluster

    cluster = LPCCondorCluster(ship_env=True)
    cluster.adapt(minimum=1, maximum=500)
    client = Client(cluster)

    for name, samples in samples2023.items():
        goldenjson = (
            "json/Cert_Collisions2023_366442_370790_Golden.json"
            if "2023" in name
            else "json/Cert_Collisions2022_355100_362760_Golden.json"
        )

        tag_n_probe = ElePt_WPTight_Gsf(
            samples,
            pt,
            goldenjson=goldenjson,
            toquery=True,
            redirect=False,
            preprocess=True,
            preprocess_args={"maybe_step_size": 100_000},
        )

        print("Starting to load events")
        tag_n_probe.load_events(from_root_args={"uproot_options": {"timeout": 120}})
        print(tag_n_probe)

        res = tag_n_probe.get_tnp_histograms(
            eta_regions_pt={
                "barrel": [0.0, 1.4442],
                "endcap_loweta": [1.566, 2.0],
                "endcap_higheta": [2.0, 2.5],
            },
            compute=True,
        )

        with uproot.recreate(f"root_files/Run{name}_Ele{pt}.root") as f:
            for var, region_dict in res.items():
                for region_name, hists in region_dict.items():
                    for i, (histname, h) in enumerate(hists.items()):
                        print(
                            f"Run{name}_Ele{pt}_{var}/{region_name}/{histname} sum : {h.sum(flow=True)}"
                        )
                        f[f"{var}/{region_name}/{histname}"] = h

        print(f"\nFinished running Run{name}_Ele{pt}\n")
