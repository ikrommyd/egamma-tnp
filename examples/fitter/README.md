Run the fitter using the example config:
```shell
run-fitter --config config.json
```
Format for the config file used for the fitter:
```
{
  "info_level": "INFO"       < ----- INFO, or DEBUG (more verbose)
  "mass": "Z",                 < ----- Determines what mass you are fitting (Z, Z_muon, JPsi, JPsi_muon)
  "input": {
    "root_files_DATA": [                                  < ----- The name will be the name of the plot file that is saved in plot_dir
        "NAME DATA 1":   ".root DATA file path 1 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
        "NAME DATA 2":   ".root DATA file path 2 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
        "NAME DATA 3":   ".root DATA file path 3 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
    ],
    "root_files_MC": [
        "NAME MC 1":     ".root MC file path 1 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
        "NAME MC 2":     ".root MC file path 2 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
        "NAME MC 3":     ".root MC file path 3 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
    ]
  },
  "fit": {
    "bin_ranges": [[5,7], [7,10], [10,20], [20,45], [45,75], [75,500]],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
    "bin": ["bin0", "bin1, etc"],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
    "fit_type": "dcb_cms"    < ----- Format is: (signal shape)_(background shape). Signal shapes: (dcb, g, dv, cbg), Background shapes: (lin, exp, cms, bpoly, cheb, ps)
    "use_cdf": false,        < ----- If a shape does not have a cdf version, defaults back to pdf
    "sigmoid_eff": false,    < ----- Switches to an unbounded efficiency that is transformed back between 0 and 1
    "interactive": true,     < ----- Turns on interactive window for fitting (very useful for difficult fits)
    "x_min": 70,             < ----- x range minimum for plotting
    "x_max": 110,            < ----- x range maximum for plotting
    "abseta": 1,             < ----- ***Only impacts muon .root files. Defines absolute eta ranges
    "numerator": "gold",     < ----- ***Only impacts muon .root files. Defines numerator for efficiencies
    "denominator": "blp"     < ----- ***Only impacts muon .root files. Defines denominator for efficiencies
  },
  "output": {
    "plot_dir": "",          < ----- Sets location to save plots to (if left blank, it won't save)
    "results_file": ""       < ----- Sets location to save results to (if left blank, it won't save)
  },
  "scale_factors": {
      "data_mc_pair": {                                      < ----- Creates explicit scale factors for pairs of data and MC files (useful for comparing one file to multiple others)
          "Scale Factor 1": ["NAME DATA 1", "NAME MC 1"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
          "Scale Factor 2": ["NAME DATA 2", "NAME MC 2"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
          "Scale Factor 3": ["NAME DATA 3", "NAME MC 3"]     < ----- Outputs scale factor of two file specified. DATA must be put before MC
    }
  }
}
```
