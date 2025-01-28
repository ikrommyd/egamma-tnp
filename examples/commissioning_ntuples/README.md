```
run-analysis \
    --config config.json \
    --settings settings.json \
    --fileset fileset.json \
    --output simplecache::root://cmseos.fnal.gov//store/user/ikrommyd/dummy/ \
    --executor dask/lpc \
    --memory 6GB \
    --scaleout 200 \
    --dashboard-address 8003 \
    --log-directory ~/dask_logs \
    --repartition-n-to-one 5 \
    --skip_report
```
