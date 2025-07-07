

Run SaS Ntuplizer on the example files (`fileset.json`)
```
run-analysis --config config.json  --fileset fileset.json  --output output/ --executor distributed --scaleout 10 --memory 20GiB
```

Get input datasets list of files from `input.txt`

```
fetch-datasets -i input.txt 
```