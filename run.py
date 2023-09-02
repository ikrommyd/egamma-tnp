import uproot
from dask.diagnostics import ProgressBar
from distributed import Client

from egamma_tnp import TagNProbe
from egamma_tnp.utils import fill_eager_pt_and_eta_histograms

if __name__ == "__main__":
    with ProgressBar():
        tag_n_probe = TagNProbe(
            [
                "/EGamma0/Run2023D-PromptReco-v1/NANOAOD",
                "/EGamma1/Run2023D-PromptReco-v1/NANOAOD",
            ],
            32,
            goldenjson="json/Cert_Collisions2023_366442_370790_Golden.json",
            toquery=True,
            redirect=False,
            preprocess=True,
            preprocess_args={"maybe_step_size": 100_000},
        )
    print("Done preprocessing")

    print("Starting to load events")
    tag_n_probe.load_events(from_root_args={"uproot_options": {"timeout": 120}})
    print(tag_n_probe)

    # from lpcjobqueue import LPCCondorCluster

    # cluster = LPCCondorCluster(ship_env=True)
    # cluster.adapt(minimum=1, maximum=100)
    # client = Client(cluster)

    client = Client()
    res = tag_n_probe.get_pt_and_eta_arrays(compute=True, scheduler=None, progress=True)

    hpt_pass, hpt_all, heta_pass, heta_all = fill_eager_pt_and_eta_histograms(res)

    print(f"Passing probes: {hpt_pass.sum(flow=True)}")
    print(f"All probes: {hpt_all.sum(flow=True)}")

    with uproot.recreate("root_files/Run2023Dv1.root") as file:
        file["hpt_pass"] = hpt_pass
        file["hpt_all"] = hpt_all
        file["heta_pass"] = heta_pass
        file["heta_all"] = heta_all
