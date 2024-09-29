```
run_analysis \
    --config config.json \
    --settings settings.json \
    --fileset fileset.json \
    --output simplecache::root://cmseos.fnal.gov//store/user/ikrommyd/dummy/ \
    --executor dask/lpc \
    --memory 6GB \
    --scaleout 200 \
    --dashboard_address 8003 \
    --log_directory ~/dask_logs \
    --repartition_n_to_one 5 \
    --skip_report
```
