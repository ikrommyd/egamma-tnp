```
run_analysis \
    --config config.json \
    --settings settings.json \
    --fileset fileset.json \
    --binning binning.json \
    --output simplecache::root://cmseos.fnal.gov//store/user/ikrommyd/dummy/ \
    --executor dask/lpc \
    --memory 3.9GB \
    --scaleout 200 \
    --dashboard_address 8003 \
    --log_directory ~/dask_logs \
    --skip_report
```
