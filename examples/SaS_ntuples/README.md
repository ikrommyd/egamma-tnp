# Running on Lxplus

Example to produce the Zee ntuples used to derive EGM Scale and Smearing corrections (https://gitlab.cern.ch/pgaigne/law_ijazz2p0).

### Activate proxy

```
voms-proxy-init --voms cms -valid 192:00
```
### Load container
```
apptainer shell -B ${XDG_RUNTIME_DIR} -B /afs -B /cvmfs/cms.cern.ch --bind /tmp  --bind /eos/user --bind /etc/sysconfig/ngbauth-submit  --env KRB5CCNAME=${XDG_RUNTIME_DIR}/krb5cc /cvmfs/unpacked.cern.ch/registry.cern.ch/cms-egamma/egamma-tnp:lxplus-el9-latest
```

#### Install local changes
```
pip install -e .
```
### Go into SaS example directory
```
cd examples/SaS_ntuples
```

### Download goldenJSON and pileup json
```
sh download_jsons.sh
```

### Get input datasets list of files to be processed

Use yaml format to set up metadata for Golden JSON and pileup reweighting

```
fetch-datasets --input input24.yaml --where Eurasia
```

### Launch SaS ntuples production using 100 jobs in htcondor (~45 minutes to process 2024 data and MC)
```
run-analysis --config config.json  --fileset input24.json  --output output --executor dask/lxplus --scaleout 100 --memory 5GiB --log-directory logs
```

Use `--executor distributed` to run interactively on the node instead of using HTCondor.
Use `run-analysis --help` to see all options.
