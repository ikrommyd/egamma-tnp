import uproot
from distributed import Client

import egamma_tnp
from egamma_tnp.triggers import ElePt_CaloIdVT_GsfTrkIdT, ElePt_WPTight_Gsf

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

mc124x_postEE = {
    "MC124x": [
        "/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
        "/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/Run3Summer22EENanoAODv10-Poisson70KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
        "/DYToLL_M-4To50_TuneCP5_13p6TeV-pythia8/Run3Summer22EENanoAODv10-Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM",
        "/DYToLL_M-4To50_TuneCP5_13p6TeV-pythia8/Run3Summer22EENanoAODv10-Poisson70KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM",
    ]
}

mc124x_preEE = {
    "MC124x_preEE": [
        "/DYTo2L_MLL-4to50_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1/NANOAODSIM",
        "/DYTo2L_MLL-50_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1/NANOAODSIM",
    ]
}

mc126x = {
    "MC126x": [
        "/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22EENanoAODv11-forPOG_126X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM"
    ]
}

mc126x_mll_50to120 = {
    "MC126x_MLL_50to120": [
        "/DYto2E_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM"
    ]
}

triggers = [ElePt_WPTight_Gsf, ElePt_CaloIdVT_GsfTrkIdT]
wptight_thresholds = [30, 32]
caloidvt_gsftrkidt_thresholds = [115, 135]
dataset_dicts = [
    samples2022,
    samples2023,
    mc124x_postEE,
    mc124x_preEE,
    mc126x,
    mc126x_mll_50to120,
]

if __name__ == "__main__":
    # client = Client()

    from lpcjobqueue import LPCCondorCluster

    cluster = LPCCondorCluster(ship_env=True)
    cluster.adapt(minimum=1, maximum=100)
    client = Client(cluster)

    for trigger in triggers:
        if trigger == ElePt_WPTight_Gsf:
            thresholds = wptight_thresholds
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
        if trigger == ElePt_CaloIdVT_GsfTrkIdT:
            thresholds = caloidvt_gsftrkidt_thresholds
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
                    200,
                    250,
                    300,
                    350,
                    400,
                ],
            )

        for pt in thresholds:
            for dataset_dict in dataset_dicts:
                for name, samples in dataset_dict.items():
                    if "2023" in name:
                        goldenjson = (
                            "json/Cert_Collisions2023_366442_370790_Golden.json"
                        )
                    elif "2022" in name:
                        goldenjson = (
                            "json/Cert_Collisions2022_355100_362760_Golden.json"
                        )
                    else:
                        goldenjson = None

                    redirect = True if "MC126x_MLL_50to120" in name else False
                    custom_redirector = (
                        "root://cmsio2.rc.ufl.edu/"
                        if "MC126x_MLL_50to120" in name
                        else "root://cmsxrootd.fnal.gov/"
                    )

                    tag_n_probe = trigger(
                        samples,
                        pt,
                        goldenjson=goldenjson,
                        toquery=True,
                        redirect=redirect,
                        custom_redirector=custom_redirector,
                        preprocess=True,
                        preprocess_args={"maybe_step_size": 100_000},
                    )

                    print("Starting to load events")
                    tag_n_probe.load_events(
                        from_root_args={"uproot_options": {"timeout": 120}}
                    )
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
