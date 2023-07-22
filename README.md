# egamma-tnp
E/Gamma High Level Trigger efficiency from NanoAOD using Tag and Probe in the [coffea](https://github.com/CoffeaTeam/coffea) framework.

## Quick Start
To use jupyter notebooks on a cluster (LPC for instance), choose a 3-digit number to replace the three instances of `xxx` below. The setup is similar for LXPLUS. We also port-forward 8787 to monitor the dask dashboard.
```bash
# connect to LPC with a port forward to access the jupyter notebook server and the dask dashboard
# remember to `kinit USERNAME@FNAL.GOV` to set up kerberos authorization before logging in
ssh USERNAME@cmslpc-sl7.fnal.gov -L8xxx:localhost:8xxx -L8787:localhost:8787
```
Then create a working directory, clone the repository and enter the directory:
```bash
cd nobackup # if this symlink does not exist, look for /uscms_data/d1/$USER
git@github.com:iasonkrom/egamma-tnp.git
cd egamma-tnp
```
The package can in principle be installed in any python virtual environment with python 3.8 or higher. That means that you can use it within a conda environment, a Singularity or Docker container, an LCG release, or any other python environment without any conflicting packages.

For simplicity, we offer a `setup.sh` bash script that gives a `coffea` Singularity shell where you can install the package and run it using the default dask configuration for LXPLUS or LPC.
```bash
bash setup.sh
```
If you are on LPC and want to use the LPC job queue, you can make use of the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue). This offers a similar setup script that gives you a `coffea` Singularity shell with the `lpcjobqueue` installed.
```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```
Both of those scripts create two new files in your directory: `shell` and `.bashrc`. The `./shell`
executable can then be used to start a Singularity shell with a `coffea` environment.
Note that the Singularity environment does inherit from your calling environment, so
it should be "clean" (i.e. no cmsenv, LCG release, etc.). For more info, refer to the README of the `lpcjobqueue` repository.

If you are on LXPLUS, you can try using a similar job queue implementation available in https://github.com/cernops/dask-lxplus but it is not recommended as LXPLUS can have unexpected reactions to the dask job queue.

Be sure your x509 grid proxy certificate is up to date before starting the shell.
```bash
voms-proxy-init --voms cms --valid 100:00
```
Once you are in your Singularity shell, you can install the `egamma-tnp` package:
```bash
pip install .
```
This works the same way in any other python virtual environment. To start the jupyter notebook, do
```bash
jupyter notebook --no-browser --port 8xxx
```
There should be a link like `http://localhost:8xxx/?token=...` displayed in the output at this point, paste that into your browser.
You should see a jupyter notebook with a directory listing.

## Running the code
A basic example of how to run the code is given in `EgammaTnP.ipynb` but here is the basic idea of how to run the code.
First you import the necessary modules:
```python
from egamma_tnp import TagNProbe
from egamma_tnp.plot import plot_efficiency
```
Then you define the `Client` that you want to use. For the default dask `Client` you would do:
```python
from distributed import Client
client = Client()
```
and to use the job queue you would do
```python
from distributed import Client
from lpcjobqueue import LPCCondorCluster

cluster = LPCCondorCluster(ship_env=True)
cluster.adapt(minimum=1, maximum=100)
client = Client(cluster)
```
More basic examples of dask client usage can be found [here](https://distributed.dask.org/en/latest/client.html).
After that, you can define a `TagNProbe` object:
```python
tag_n_probe = TagNProbe(
    [
        "/EGamma0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
        "/EGamma1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
    ],
    32,
    goldenjson="json/Cert_Collisions2023_366442_368823_Golden.json",
    toquery=True,
    redirect=False,
)
```
Please refer to its docstring for more information on the arguments.
Then you can load the events using
```python
tag_n_probe.load_events()
```
and then perform tag and probe to get the $P_T$ and $\eta$ histograms of the passing and all probes:
```python
(
    hpt_pass,
    hpt_all,
    heta_pass,
    heta_all,
) = tag_n_probe.get_tnp_histograms(compute=True, scheduler=None)
```
You can do whatever you want with those histograms but to plot the efficiencies as a function of $P_T$ and $\eta$ you would do:
```python
plot_efficiency(hpt_pass, hpt_all)
plot_efficiency(heta_pass, heta_all)
```
The customization of those plots is up to the user. Using [mplhep](https://github.com/scikit-hep/mplhep) is recommended.
