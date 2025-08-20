

Run SaS Ntuplizer on the example files (`fileset.json`)
```
run-analysis --config config.json  --fileset fileset.json  --output output/ --executor distributed --scaleout 10 --memory 5GiB
```

Get input datasets list of files from `input.txt`

```
fetch-datasets -i input.txt 
```

Get input datasets list of files from `inputDAS.yaml` to set up metadata for GoldenJSON and pileup reweighting

```
fetch-datasets -i inputDAS.yaml
```