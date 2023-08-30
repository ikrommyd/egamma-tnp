import uproot
from distributed import Client

from egamma_tnp import TagNProbe

if __name__ == "__main__":
    client = Client()

    tag_n_probe = TagNProbe(
        [
            "/EGamma0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
            "/EGamma1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        ],
        32,
        goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
        toquery=True,
        redirect=False,
        preprocess=True,
        preprocess_args={"maybe_step_size": 100_000},
    )

    # tag_n_probe.remove_bad_xrootd_files(
    #     [
    #         "root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/data/Run2023C/EGamma0/NANOAOD/PromptNanoAODv12_v3-v1/80000/0a9e4a46-1546-4af0-ba06-73da06adab06.root",
    #     ]
    # )
    # tag_n_probe.redirect_files(
    #     [
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma0/NANOAOD/PromptNanoAODv11p9_v1-v1/70000/a4252c75-26ad-4c46-b8dd-ac8b49b4cb68.root",
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma1/NANOAOD/PromptNanoAODv11p9_v1-v1/70000/a22f4952-e686-46d8-8209-ec2602cf6b6d.root",
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma1/NANOAOD/PromptNanoAODv11p9_v1-v1/2810000/67696448-126e-4ad8-83dd-4de09b867c8c.root",
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma1/NANOAOD/PromptNanoAODv11p9_v1-v1/2820000/6aa674c1-803c-48fa-babb-33319a35d3ac.root",
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma1/NANOAOD/PromptNanoAODv11p9_v1-v1/2820000/1e3c8eb9-329a-4a47-ab4d-b1b2f0f8fe9f.root",
    #         "root://cmsxrootd.fnal.gov//store/data/Run2023B/EGamma0/NANOAOD/PromptNanoAODv11p9_v1-v1/70000/ec91ee77-3fca-4994-b27f-1f4894423044.root",
    #     ],
    #     redirectors="root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/",
    # )

    tag_n_probe.load_events()
    print(tag_n_probe)

    (
        hpt_pass,
        hpt_all,
        heta_pass,
        heta_all,
    ) = tag_n_probe.get_tnp_histograms(compute=True, scheduler=None, progress=True)

    print(f"Passing probes: {hpt_pass.sum(flow=True)}")
    print(f"All probes: {hpt_all.sum(flow=True)}")

    with uproot.recreate("root_files/Run2023Cv3.root") as file:
        file["hpt_pass"] = hpt_pass
        file["hpt_all"] = hpt_all
        file["heta_pass"] = heta_pass
        file["heta_all"] = heta_all
