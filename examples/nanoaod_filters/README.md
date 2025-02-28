```
run-analysis \
    --config config1/2.json \
    --settings settings.json \
    --fileset fileset.json \
    --binning binning.json \
    --output simplecache::root://cmseos.fnal.gov//store/user/ikrommyd/dummy/ \
    --executor dask/lpc \
    --memory 3.9GB \
    --scaleout 200 \
    --dashboard-address 8003 \
    --log-directory ~/dask_logs \
    --skip-report
```
