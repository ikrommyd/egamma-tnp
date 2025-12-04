# egamma-tnp

[![ci](https://github.com/ikrommyd/egamma-tnp/actions/workflows/ci.yml/badge.svg)](https://github.com/ikrommyd/egamma-tnp/actions?query=workflow%3ACI%2FCD+event%3Aschedule+branch%3Amaster)
[![PyPI Version](https://badge.fury.io/py/egamma-tnp.svg)](https://badge.fury.io/py/egamma-tnp)
![Downloads](https://img.shields.io/pypi/dm/egamma-tnp.svg)

E/Gamma Tag & Probe framework using [coffea](https://github.com/CoffeaTeam/coffea).

# Introduction

This framework is purely Python, so one just needs a python environment to run it.
The code is documented  with docstrings.
One can read the source, use the `help` method, or just hover over the docstrings in their favorite Python IDE, Jupyter etc.

Recommended way to work:
* Use the CLI if you don't plan to change the code.
* If you want to make changes to the code, you can use the programmatic API of the package. To see how the API maps to the CLI, you can look at `tests/test_cli.py` which does just that manually.
* Finally, you can adapt the CLI to reflect the changes you made to the code.

NOTE: the documentation of the programmatic API is a TODO, but the docstrings should help you get started.

There are some examples in the `examples` folder that more or less show you how to configure the CLI.

# Installation

## Installation to just run the framework

The framework is a pure python package, so one just needs a python environment to be able to run it.
The framework is available on PyPI, so one can install it via `pip` (or similar like `uv`):
```
pip install egamma-tnp
```
We also provide a ready-to-use docker container that has the package pre-installed along with all dependencies.
You can run the container with something like this (or similar):
```
docker run --rm -it registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest /bin/bash
```
This will get you the package as is in master branch currently. You can also choose a specific version by replacing `latest` with the version tag you want like `lxplus-el9-v0.5.0` for example.

Finally, the docker containers get unpacked as apptainer/singularity images in CVMFS, so if you are on lxplus or any other place that has CVMFS mounted, you can get a shell in the apptainer image like this:
```
apptainer shell -B ${XDG_RUNTIME_DIR} -B /afs -B /cvmfs/cms.cern.ch --bind /tmp  --bind /eos/user/<initial>/<username> --bind /etc/sysconfig/ngbauth-submit  --env KRB5CCNAME=${XDG_RUNTIME_DIR}/krb5cc /cvmfs/unpacked.cern.ch/registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest
```

## Installation for development

To set up a development environment for the framework, can can again use any python environment that lets you install packages in editable mode.

You can then clone the repository and install it in editable mode with development dependencies:
```
git clone https://github.com/ikrommyd/egamma-tnp.git # or your fork
cd egamma-tnp/
pip install -e '.[dev]'
```
If you are installing in an environment where you can't modify the packages of the environment, the package will be installed as a user package (that can also be done explicitly with `--user` flag):
```
pip install --user -e '.[dev]'
```
In that case, the command line tools the package provides will be installed in `~/.local/bin`, so make sure that this folder is in your `PATH`.
```
export PATH="$HOME/.local/bin:$PATH"
```

You can then run the tests to make sure everything is working:
```
python -m pytest tests
```
You can also check that the CLI is there which you should see after installation with something like:
```
run-analysis --help
```
To test the CLI you can explicitly run its test:
```
python -m pytest tests/test_cli.py -v
```

# Inputs needed to run the framework

To run the framework, we you need a few JSON files to define input files, what to run from the framework, selections and other options, and binning. Example files are included in the `tests` folder and also in the `examples` folder. For example, these are the files the CLI test uses:
```
tests/example_fileset.json
tests/example_runner.json
tests/example_settings.json
tests/example_binning.json
tests/output/ (output folder)
```
### The fileset JSON
The fileset JSON defines the input datasets, their root files, and some metadata like whether the dataset is MC or data, golden JSON for data, pileup weights for MC etc.
An example of such a structure can be seen below:
```json
{
  "Data": {
    "files": {
      "File1.root": "Events",
      "File2.root": "Events"
    },
    "metadata": {
      "isMC": false,
      "goldenJSON": "Cert_Collisions2024_378981_386951_Golden.json"
    }
  },
  "MC": {
    "files": {
      "File1.root": "Events",
      "File2.root": "Events"
    },
    "metadata": {
      "isMC": true,
      "pileupJSON": "puWeights_2024.json.gz"
    }
  }
}
```
For quick testing, you can use the provided samples under `tests/samples`. If one needs more or larger files for local testing, you can download files to your machine with xrootd:
```
xrdcp root://cms-xrd-global.cern.ch///<path/to/file>.root .
```
and then use them as input files in the fileset JSON.

### The runner JSON
Defines the filters (cuts) and tasks that produce histograms/outputs.
Example filter block (probe electron):
```json
{
  "filters": {
    "HLT_Ele30_WPTight_Gsf": "HLT_Ele30_WPTight_Gsf",
    "cutBased >= 2": "cutBased >= 2"
  }
}
```
This means that we will check the efficiency against two filters. One is where the a probe is considered passing if it fired the HLT_Ele30_WPTight_Gsf trigger, and the other is where the probe is considered passing if its cutBased ID is at least 2.
Once defined, pass these filters into methods that produce histograms or NTuples like so in the same JSON:
```json
{
  "name": "get_nd_tnp_histograms",
  "args": {
    "filter": [
      "HLT_Ele30_WPTight_Gsf",
      "cutBased >= 2"
    ],
    "cut_and_count": false,
    "vars": [
      "el_pt",
      "el_eta"
    ],
    "uproot_options": {
      "allow_read_errors_with_report": [
        "OSError",
        "ValueError"
      ],
      "timeout": 120
    }
  }
}
```
You can see more examples of such runner JSONs in the `examples` folder. You will also see that there is a `workflow` field in the runner JSONs. That defines which workflow to run. Currently available workflows are
```
ElectronTagNProbeFromMiniNTuples
ElectronTagNProbeFromNanoAOD
ElectronTagNProbeFromNanoNTuples
PhotonTagNProbeFromMiniNTuples
PhotonTagNProbeFromNanoAOD
PhotonTagNProbeFromNanoNTuples
ScaleAndSmearingNTuplesFromNanoAOD
```
All the options the runner and settings JSONs map to class initialization arguments while the method arguments map to inputs to the methods defined by their parent `BaseTagNProbe` or `BaseNTuplizer` classes.
So look at the docstrings of those classes and methods to learn more about what options are available and what they do.

### The settings JSON
Holds extra analysis-level cuts. For example: pT and other selections on tag and probe leptons etc.

### The binning JSON
Defines the binning used in the analysis (e.g., for el_pt, el_eta). Defaults are provided, but one can change them.

### The output
The output folder contains the result of applying each method to all datasets in the fileset JSON and aggregating the results per dataset. For NTuplization workflows, parquet files will be produced. For histogramming workflows, pickle files will be produced that contain the histograms.

# Run it

Once you have have all the ingredients, you can run the framework using the CLI like so:

```
run-analysis --config tests/example_runner.json --settings tests/example_settings.json --fileset tests/example_fileset.json --binning tests/example_binning.json --output tests/output --executor threads
```
Look at `run-analysis --help` for all available options.

If everything works, you’ll find .pkl files in `tests/output/`.
Feel free to explore the output. Histograms are `hist.Hist` objects from the `hist` package and therefore fully support [uhi](https://uhi.readthedocs.io/en/latest/).
Look at the [`uhi` docs](https://uhi.readthedocs.io/en/latest/) and [`hist` docs](https://hist.readthedocs.io/en/latest/) for more information on how to manipulate and plot these histograms.
For example in python you can do:
```python
import pickle
from matplotlib import pyplot as plt

with open("<filename>.pkl", "rb") as f:
       data = pickle.load(f)

data["passing"].project["mll"].plot()
plt.show()
```

# Pre-processing

Before running the complete analysis, it’s important to pre-process the data. Typical tasks include:
1. Cleaning the input file lists (remove unreadable files)
2. Splitting files into chunks of roughly equal size for better parallel processing. This also ensures that the chunks can fit into the workers' memory.

In order to do this, prepare a JSON listing of the files to process. One can split this per era(recommended).
### Make a dataset name to DAS dataset name mapping
Create a simple text file (e.g., `samples.txt`) with one dataset per line: For example if the year is 2024:
```
DataC_2024 /EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD
DataC_2024 /EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD
```

### Get the list of files for each dataset
Use `fetch-datasets` to resolve each dataset into full file paths. Choose an xrootd redirector via the `--where` option. Available redirectors are
```
"Americas": "root://cmsxrootd.fnal.gov/"
"Eurasia": "root://xrootd-cms.infn.it/"
"Yolo": "root://cms-xrd-global.cern.ch/"
```
Run the fetching like so:
```
fetch-datasets  --input samples.txt --where Eurasia
```

This generates a `<txt file name>.json` with the full list of files per dataset that you can feed into pre-processing.

### Run the pre-processing step
For the pre-processing one can check the coffea documentation of the function [`coffea.dataset_tools.preprocess`](https://coffea-hep.readthedocs.io/en/latest/api/coffea.dataset_tools.preprocess.html#coffea.dataset_tools.preprocess) for all available options.
If anyone requires, an example for the pre-processing, it is available here: https://github.com/MGhimiray/egamma-tnp/blob/dev_promptMVA/promptMVA/Dataset/preprocess.py.

The preprocessing produces to JSON-serializable dictionaries. A fileset that contains all the "runnable" chunks (i.e. those that it was able to read) and a "processed" fileset which contains all the chunks and has `None`s for the unreadable files chunks specifications. You mostly care about the former. Those filesets can be dumped into JSON files and used as input to the main analysis. You can keep them forever as long as the files that they point to have not changed.

# Running full analysis

To run the full analysis one can make use of the container. Before doing that remember to use voms cert. with the command:
```
voms-proxy-init -voms cms
```
If you are using the apptainer image, you need to do this before firing up the shell.
To run the analysis you can do something like this:
```
run-analysis --config runner.json --settings settings.json --fileset fileset.json --binning binning.json --output simplecache::root://eosuser.cern.ch//eos/user/<specify_user>/foldername/ --executor dask/lxplus --memory 8GB --scaleout 100 --dashboard-address 8002 --log-directory /eos/user/<specify_user>/condor/log --queue espresso
```
To be able to take a look at the dask dashboard while the jobs are running, you can port-forward a port during your ssh login like so `ssh -L 8xxx:localhost:8xxx <your_lxplus_username>@lxplus.cern.ch` and then you can specify the same port in the `--dashboard-address` argument.

The execution is similar in other clusters but the way of manage environments and also connect to workers via a dask client is different per cluster so please consult your analysis facility's documentation. For example, on FNAL LPC, you need to use `lpcjobqueue` to connect to the workers: https://github.com/CoffeaTeam/lpcjobqueue.


If you do not care to check which files you failed to access and get a report back regarding file access during execution and errors, you can also pass `--skip-report` to not compute the report. The report is otherwise a json file that is dumped in the output folders.

NOTE for lxplus + apptainer: If workers don’t appear, check that no stray Python/Dask processes from previous runs are still running. Use `ps aux` and `grep` for the `run-analysis` jobs and terminate them if needed.

This often happens when a job is interrupted with Ctrl+C. Dask clusters are asynchronous and can linger briefly unless shut down explicitly.

Hint: If you are having trouble with keeping the ssh session alive during the execution, you can make use of tmux. The instruction for doing this in lxplus is:
```
systemctl --user start tmux.service
tmux attach
```

Make a note of the LXPLUS node where the tmux session is running, typically by typing `hostname`. You need to reconnect to the same node later to find your tmux session.
