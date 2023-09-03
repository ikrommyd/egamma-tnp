import uproot

from egamma_tnp import TagNProbe, utils

if __name__ == "__main__":
    era = "2023C"
    datasets = [
        "/EGamma0/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv11p9_v1-v1/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        "/EGamma0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
    ]

    failed_files = []
    exceptions = []

    for i, dataset in enumerate(datasets):
        files = utils.get_dataset_files_replicas(dataset, mode="first")[0]
        for j, file in enumerate(files):
            # file = utils.redirect_files(file, isrucio=True).pop()
            print(f"Running on {file}")
            tag_n_probe = TagNProbe(
                file,
                32,
                goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
                toquery=False,
                redirect=False,
            )

            try:
                tag_n_probe.load_events(
                    from_root_args={
                        "uproot_options": {"timeout": 120},
                        "chunks_per_file": 10,
                    }
                )

                (
                    hpt_pass,
                    hpt_all,
                    heta_pass,
                    heta_all,
                ) = tag_n_probe.get_tnp_histograms(
                    compute=True, scheduler="processes", progress=True
                )

                print(f"Passing probes: {hpt_pass.sum(flow=True)}")
                print(f"All probes: {hpt_all.sum(flow=True)}")

                with uproot.recreate(
                    f"root_files/Run{era}_EGamma{i}_file{j}.root"
                ) as file:
                    file["hpt_pass"] = hpt_pass
                    file["hpt_all"] = hpt_all
                    file["heta_pass"] = heta_pass
                    file["heta_all"] = heta_all

            except OSError as e:
                print(f"{file} failed")
                print(e)
                failed_files.append(file)
                exceptions.append(e)

    with open("failed_files.txt", "w") as f:
        for filename, exc in zip(failed_files, exceptions):
            f.write(f"{filename}\t{exc}\n")
