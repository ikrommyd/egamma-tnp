# egamma-tnp

[![ci](https://github.com/ikrommyd/egamma-tnp/actions/workflows/ci.yml/badge.svg)](https://github.com/ikrommyd/egamma-tnp/actions?query=workflow%3ACI%2FCD+event%3Aschedule+branch%3Amaster)
[![PyPI Version](https://badge.fury.io/py/egamma-tnp.svg)](https://badge.fury.io/py/egamma-tnp)
![Downloads](https://img.shields.io/pypi/dm/egamma-tnp.svg)

E/Gamma Tag & Probe framework using [coffea](https://github.com/CoffeaTeam/coffea).# egamma-tnp

[![ci](https://github.com/ikrommyd/egamma-tnp/actions/workflows/ci.yml/badge.svg)](https://github.com/ikrommyd/egamma-tnp/actions?query=workflow%3ACI%2FCD+event%3Aschedule+branch%3Amaster)
[![PyPI Version](https://badge.fury.io/py/egamma-tnp.svg)](https://badge.fury.io/py/egamma-tnp)
![Downloads](https://img.shields.io/pypi/dm/egamma-tnp.svg)

E/Gamma Tag & Probe framework using [coffea](https://github.com/CoffeaTeam/coffea).

# Introduction

This framework is purely Python, so one needs Python env. The code is documented end-to-end. One can read the source or just hover over the docstrings in Python IDE / Jupyter etc. 

Recommended way to work:
1) Use the CLI if one doesn't plan to change the code.
2) If one changes the code, update the CLI to make them usable there.

NOTE: any python environment will do fine. pip install egamma-tnp is available from pypi if anyone doesn't plan to make any changes to the code and don't want to run tests etc.  
There are a few ready-to-run examples here (CLI): [examples/](https://github.com/ikrommyd/egamma-tnp/tree/master/tests).  

# Installations
First install a few packages, clone the repo, and run a couple of quick checks.  
1) Choose where to work (local or lxplus)  
One can do this on local machine laptop/PC or on lxplus. Steps are the same.
2) Clone the repo and install:
   
       git clone https://github.com/ikrommyd/egamma-tnp.git
       cd egamma-tnp/
       pip install -e '.[dev]'
       export PATH="$HOME/.local/bin:$PATH"
3) Quick checks  
Check if everything went well:

       python -V
       python -m pytest --version
       python -c "import egamma_tnp; print(egamma_tnp.__version__)"
Check the command-line tool:

       run-analysis --help
 Optionally, one can also check if the CLI is working as expected by running a simple command:

       pytest tests/test_cli.py -v 
If its fine the output should give:

      tests/test_cli.py::test_cli PASSED 

# Inputs needed to run the framework

To run the framework, we use a few JSON files to define input files, cuts, and binning. Example files are included in the repo:  
tests/example_fileset.json  
tests/example_runner.json  
tests/example_settings.json  
tests/example_binning.json  
tests/output/ (output folder)  

1) tests/example_fileset.json  
Lists the ROOT files used for a given analysis. The repo already includes a small test sample:

tests/samples/DYto2E.root
One can also point to:  
    the golden JSON for data,
    pileup weights for MC.

Example: 
    
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

For quick testing, use the provided sample. If one needs more files, copy via xrootd:

       xrdcp root://cms-xrd-global.cern.ch///<path/to/file>.root .  
and then specify the local path in the JSON.

2) tests/example_runner.json  
Defines the filters (cuts) and tasks that produce histograms/outputs.  
Example filter block (probe electron):

       {
         "filters": {
           "HLT_Ele30_WPTight_Gsf": "HLT_Ele30_WPTight_Gsf",
           "cutBased >= 2": "cutBased >= 2"
         }
       }

Once defined, pass these filters to the histogram task:

      {
         "name": "get_nd_tnp_histograms",
         "args": {
           "filter": [
             "HLT_Ele30_WPTight_Gsf",
             "cutBased >= 2"
           ],
           "cut_and_count": false,
           "vars": ["el_pt", "el_eta"],
           "uproot_options": {
             "allow_read_errors_with_report": ["OSError", "ValueError"],
             "timeout": 120
           }
         }
       }

3) tests/example_settings.json  
Holds extra analysis-level cuts. For example: pT and η selections on tag and probe leptons etc.

4) tests/example_binning.json  
Defines the binning used in the analysis (e.g., for el_pt, el_eta). Defaults are provided, but one can change them.

5) tests/output/  
Output folder for histograms (pickled .pkl files).  
Note: Unit tests may delete outputs to keep the tree clean. If one wants to keep test outputs, comment out this line in  
tests/test_cli.py:

    os.system("rm -r tests/output")

# Run it

After editing the JSONs as you like, run:

       run-analysis --config tests/example_runner.json --settings tests/example_settings.json --fileset tests/example_fileset.json --binning tests/example_binning.json --output tests/output --executor threads

If everything works, you’ll find .pkl files in tests/output/.  
To read the .pkl files, see the indexing guide here: [UHI docs — indexing/](https://uhi.readthedocs.io/en/latest/indexing.html).  
Or a simple way to do this would be:

       python3
       import pickle
       from matplotlib import pyplot as plt
       with open("<filename>.pkl", "rb") as f:
           data = pickle.load(f)
       data   # this will print the contents of .pkl file
       data["passing"].project["mll"]  # any passing or failing histograms can be plotted in this way

# Pre-processing

Before running the complete analysis, it’s important to pre-process the data. Typical tasks include:  
Cleaning the input lists (remove bad files)  
Splitting large files into smaller chunks  
Ensures each chunk fits in worker RAM  

Pre-processing helps one catch problems early before scaling up. For this, prepare a JSON listing of the files to process. One can split this per era(recommended) or do it year wise.  
1) Make a dataset list  
Create a simple text file (e.g., samples.txt) with one dataset per line: For e.x if the year is 2024:

       DataC_2024 /EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD
       DataC_2024 /EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD

2) Fetch full dataset info  
Use fetch_datasets.py to resolve each dataset into full file paths. Choose an xrootd redirector via --where:

        fetch_datasets.py  --input samples.txt --where Eurasia

This generates a .json with the full dataset information you’ll feed into pre-processing.  

       Available redirectors (choose one):
       "Americas": "root://cmsxrootd.fnal.gov/"
       "Eurasia": "root://xrootd-cms.infn.it/"
       "Yolo": "root://cms-xrd-global.cern.ch/"

3) Run the pre-processor  
For the pre-processing one can check the coffea docs: [coffea-hep/](https://coffea-hep.readthedocs.io/en/latest/api/coffea.dataset_tools.preprocess).  
If anyone requires, an example for the pre-processing, it is available here: [preprocess/](https://github.com/MGhimiray/egamma-tnp/blob/dev_promptMVA/promptMVA/Dataset/preprocess.py).  
Inside this example preprocess.py, point FILES to your JSON(s):
 
       FILES = [
         "DataC.json"
       ]
The pre-processing produces two lists: processed (summary of how many files were read vs. failed) and runnable (the clean list you will actually use for analysis). 

# Running full analysis

To run the full analysis one can make use of the container. Before doing that remember to use voms cert. with the command:

       voms-proxy-init -voms cms
         
For the container we need: (remember to specify user name)

           apptainer shell -B ${XDG_RUNTIME_DIR} -B /afs -B /cvmfs/cms.cern.ch --bind /tmp  --bind /eos/user/<specify_user> --bind /etc/sysconfig/ngbauth-submit  --env KRB5CCNAME=${XDG_RUNTIME_DIR}/krb5cc /cvmfs/unpacked.cern.ch/registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest
         
Finally use the run-analysis with singularity:
        
           run-analysis --config runner.json --settings settings.json --fileset fileset.json --binning binning.json --output simplecache::root://eosuser.cern.ch//eos/user/<specify_user>/foldername/ --executor dask/lxplus --memory 8GB --scaleout 100 --dashboard-address 8002 --log-directory /eos/user/<specify_user>/condor/log --queue espresso
         
One needs to define the job --queue from any one of the following:

       espresso = 20 minutes
       microcentury = 1 hour
       longlunch = 2 hours
       workday = 8 hours
       tomorrow = 1 day
       testmatch = 3 days
       nextweek = 1 week

If it is not required to check the entire report of the analysis and if anyone wants to rapidly complete the process, we can skip printing the report using
       --skip-report  
NOTE: If workers don’t appear, check that no stray Python/Dask processes from previous runs are still running. Use ps aux and grep for the run-analysis jobs then terminate them if needed.

This often happens when a job is interrupted with Ctrl+C. Dask clusters are asynchronous and can linger briefly unless shut down explicitly. To stop them cleanly, call
        cluster.shutdown()  
Hint: If anyone is having trouble with keeping the terminal alive during the task, we can make use of tmux. The instruction for doing this in lxplus is:

       Start a session: systemctl --user start tmux.service
       Attach: tmux attach

Make a note of the LXPLUS node where the tmux session is running, typically by typing hostname—it will be required to reconnect later. If required have a look at the tmux documentation, [tmux/](https://cern.service-now.com/service-portal?id=kb_article&n=KB0008111)  
Start the apptainer and rest will remain same. 

