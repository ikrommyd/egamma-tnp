import getpass
import json
import os
import re
import subprocess
from collections import defaultdict

import numpy as np
from hist import intervals
from rucio.client import Client

# Rucio needs the default configuration --> taken from CMS cvmfs defaults
os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/current"


def merge_goldenjsons(files, outfile):
    """Merge multiple golden jsons into one.

    Parameters
    ----------
        files : list of str
            The list of golden jsons to merge.
        outfile : str
            The output file path.
    """
    dicts = []
    for file in files:
        with open(file) as f:
            dicts.append(json.load(f))

    output = {}
    for d in dicts:
        for key, value in d.items():
            if key in output and isinstance(output[key], list):
                # if the key is in the merged dictionary and its value is a list
                for item in value:
                    if item not in output[key]:
                        # if the value is not in the list of values for the key in output, append it
                        output[key].append(item)
            else:
                # otherwise, add the key and value to the merged dictionary
                output[key] = value

    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)


def replace_nans(arr):
    arr = np.array(arr)

    # Find the index of first non-nan value
    first_float_index = np.where(~np.isnan(arr))[0][0]

    # Create masks for before and after the first float
    before_first_float = np.arange(len(arr)) < first_float_index
    after_first_float = ~before_first_float

    # Replace all nans with 0 before first float number and with 1 after
    arr[before_first_float & np.isnan(arr)] = 0
    arr[after_first_float & np.isnan(arr)] = 1

    return arr


def get_proxy_path():
    """
    Checks if the VOMS proxy exists and if it is valid
    for at least 1 hour.
    If it exists, returns the path of it"""
    try:
        subprocess.run("voms-proxy-info -exists -hours 1", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expirend or non-existing: please run `voms-proxy-init -voms cms -rfc --valid 168:0`"
        )

    # Now get the path of the certificate
    proxy = subprocess.check_output(
        "voms-proxy-info -path", shell=True, text=True
    ).strip()
    return proxy


def get_rucio_client(proxy=None):
    """
    Open a client to the CMS rucio server using x509 proxy.
    Parameters
    ----------
        proxy : str, optional
            Use the provided proxy file if given, if not use `voms-proxy-info` to get the current active one.
    Returns
    -------
        nativeClient: rucio.Client
            Rucio client
    """
    try:
        if not proxy:
            proxy = get_proxy_path()

        nativeClient = Client(
            rucio_host="https://cms-rucio.cern.ch",
            auth_host="https://cms-rucio-auth.cern.ch",
            account=getpass.getuser(),
            creds={"client_cert": proxy, "client_key": proxy},
            auth_type="x509",
        )
        return nativeClient

    except Exception as e:
        print("Wrong Rucio configuration, impossible to create client")
        raise e


def get_xrootd_sites_map():
    """
    The mapping between RSE (sites) and the xrootd prefix rules is read
    from `/cvmfs/cms/cern.ch/SITECONF/*site*/storage.json`.
    This function returns the list of xrootd prefix rules for each site.
    """
    sites_xrootd_access = defaultdict(dict)
    # TODO Do not rely on local sites_map cache. Just reload it?
    if not os.path.exists(".sites_map.json"):
        print("Loading SITECONF info")
        sites = [
            (s, "/cvmfs/cms.cern.ch/SITECONF/" + s + "/storage.json")
            for s in os.listdir("/cvmfs/cms.cern.ch/SITECONF/")
            if s.startswith("T")
        ]
        for site_name, conf in sites:
            if not os.path.exists(conf):
                continue
            try:
                data = json.load(open(conf))
            except Exception:
                continue
            for site in data:
                if site["type"] != "DISK":
                    continue
                if site["rse"] is None:
                    continue
                for proc in site["protocols"]:
                    if proc["protocol"] == "XRootD":
                        if proc["access"] not in ["global-ro", "global-rw"]:
                            continue
                        if "prefix" not in proc:
                            if "rules" in proc:
                                for rule in proc["rules"]:
                                    sites_xrootd_access[site["rse"]][
                                        rule["lfn"]
                                    ] = rule["pfn"]
                        else:
                            sites_xrootd_access[site["rse"]] = proc["prefix"]
        json.dump(sites_xrootd_access, open(".sites_map.json", "w"))

    return json.load(open(".sites_map.json"))


def _get_pfn_for_site(path, rules):
    """
    Utility function that converts the file path to a valid pfn matching
    the file path with the site rules (regexes).
    """
    if isinstance(rules, dict):
        for rule, pfn in rules.items():
            if m := re.match(rule, path):
                grs = m.groups()
                for i in range(len(grs)):
                    pfn = pfn.replace(f"${i+1}", grs[i])
                return pfn
    else:
        return rules + "/" + path


def get_dataset_files_replicas(
    dataset,
    whitelist_sites=None,
    blacklist_sites=None,
    regex_sites=None,
    mode="full",
    client=None,
):
    """
    This function queries the Rucio server to get information about the location
    of all the replicas of the files in a CMS dataset.
    The sites can be filtered in 3 different ways:
    - `whilist_sites`: list of sites to select from. If the file is not found there, raise an Exception.
    - `blacklist_sites`: list of sites to avoid. If the file has no left site, raise an Exception
    - `regex_sites`: regex expression to restrict the list of sites.
    The fileset returned by the function is controlled by the `mode` parameter:
    - "full": returns the full set of replicas and sites (passing the filtering parameters)
    - "first": returns the first replica found for each file
    - "best": to be implemented (ServiceX..)
    - "roundrobin": try to distribute the replicas over different sites
    Parameters
    ----------
        dataset: str
        whilelist_sites: list
        blacklist_sites: list
        regex_sites: list
        mode:  str, default "full"
        client: rucio Client, optional
    Returns
    -------
        files: list
           depending on the `mode` option.
           - If `mode=="full"`, returns the complete list of replicas for each file in the dataset
           - If `mode=="first"`, returns only the first replica for each file.
        sites: list
           depending on the `mode` option.
           - If `mode=="full"`, returns the list of sites where the file replica is available for each file in the dataset
           - If `mode=="first"`, returns a list of sites for the first replica of each file.
        sites_counts: dict
           Metadata counting the coverage of the dataset by site
    """
    sites_xrootd_prefix = get_xrootd_sites_map()
    client = client if client else get_rucio_client()
    outsites = []
    outfiles = []
    for filedata in client.list_replicas([{"scope": "cms", "name": dataset}]):
        outfile = []
        outsite = []
        rses = filedata["rses"]
        found = False
        if whitelist_sites:
            for site in whitelist_sites:
                if site in rses:
                    # Check actual availability
                    meta = filedata["pfns"][rses[site][0]]
                    if (
                        meta["type"] != "DISK"
                        or meta["volatile"]
                        or filedata["states"][site] != "AVAILABLE"
                        or site not in sites_xrootd_prefix
                    ):
                        continue
                    outfile.append(
                        _get_pfn_for_site(filedata["name"], sites_xrootd_prefix[site])
                    )
                    outsite.append(site)
                    found = True

            if not found:
                raise Exception(
                    f"No SITE available in the whitelist for file {filedata['name']}"
                )
        else:
            possible_sites = list(rses.keys())
            if blacklist_sites:
                possible_sites = list(
                    filter(lambda key: key not in blacklist_sites, possible_sites)
                )

            if len(possible_sites) == 0:
                raise Exception(f"No SITE available for file {filedata['name']}")

            # now check for regex
            for site in possible_sites:
                if regex_sites:
                    if re.search(regex_sites, site):
                        # Check actual availability
                        meta = filedata["pfns"][rses[site][0]]
                        if (
                            meta["type"] != "DISK"
                            or meta["volatile"]
                            or filedata["states"][site] != "AVAILABLE"
                            or site not in sites_xrootd_prefix
                        ):
                            continue
                        outfile.append(
                            _get_pfn_for_site(
                                filedata["name"], sites_xrootd_prefix[site]
                            )
                        )
                        outsite.append(site)
                        found = True
                else:
                    # Just take the first one
                    # Check actual availability
                    meta = filedata["pfns"][rses[site][0]]
                    if (
                        meta["type"] != "DISK"
                        or meta["volatile"]
                        or filedata["states"][site] != "AVAILABLE"
                        or site not in sites_xrootd_prefix
                    ):
                        continue
                    outfile.append(
                        _get_pfn_for_site(filedata["name"], sites_xrootd_prefix[site])
                    )
                    outsite.append(site)
                    found = True

        if not found:
            raise Exception(f"No SITE available for file {filedata['name']}")
        else:
            if mode == "full":
                outfiles.append(outfile)
                outsites.append(outsite)
            elif mode == "first":
                outfiles.append(outfile[0])
                outsites.append(outsite[0])
            else:
                raise NotImplementedError(f"Mode {mode} not yet implemented!")

    # Computing replicas by site:
    sites_counts = defaultdict(int)
    if mode == "full":
        for sites_by_file in outsites:
            for site in sites_by_file:
                sites_counts[site] += 1
    elif mode == "first":
        for site_by_file in outsites:
            sites_counts[site] += 1

    return outfiles, outsites, sites_counts


def query_dataset(query, client=None, tree=False):
    client = client if client else get_rucio_client()
    out = list(
        client.list_dids(
            scope="cms", filters={"name": query, "type": "container"}, long=False
        )
    )
    if tree:
        outdict = {}
        for dataset in out:
            split = dataset[1:].split("/")
            if split[0] not in outdict:
                outdict[split[0]] = defaultdict(list)
            outdict[split[0]][split[1]].append(split[2])
        return out, outdict
    else:
        return out


def dasgoclient_query(dataset, *, invalid=False):
    """Get the list of files from DAS for the given dataset.

    Parameters
    ----------
        dataset : str
            The dataset name to query for its files.
        invalid : bool, optional
            Whether to include invalid files. The default is False.

    Returns
    -------
        files: list of str
            The list of files.
    """
    files = []
    if invalid:
        files.extend(
            os.popen(f"dasgoclient --query='file dataset={dataset} status=*'")
            .read()
            .splitlines()
        )
    else:
        files.extend(
            os.popen(f"dasgoclient --query='file dataset={dataset}'")
            .read()
            .splitlines()
        )
    return files


def redirect_files(files, *, redirector="root://cmsxrootd.fnal.gov/", isrucio=False):
    """Add an xrootd redirector to a list of files

    Parameters
    ----------
        files : str or list of str
            The list of files to redirect.
        redirector : str, optional
            The xrootd redirector to add. The default is "root://cmsxrootd.fnal.gov/".
        isrucio : bool, optional
            Whether the files were queried from rucio. The default is False.
    """
    if isinstance(files, str):
        files = [files]

    if isrucio:
        return [redirector + "/store/" + file.split("/store/")[1] for file in files]
    else:
        return [redirector + file for file in files]


def get_file_dict(datasets, *, custom_redirector=None, invalid=False):
    """Get the lists of files from DAS for the given dataset names.
    The list of files is returned as a dictionary with the dataset names as keys
    and the lists of files as values.

    Parameters
    ----------
        datasets : str or list of str
            The dataset names to query.
        custom_redirector : str, optional
            The xrootd redirector to add to the files. The default is None.
            If None, this function will query rucio and add the redirector for the first available site.
        invalid : bool, optional
            Whether to include invalid files. The default is False.
            A custom redirector must be provided if invalid is True.

    Returns
    -------
        file_dict: dict
            A dictionary of {dataset : files} pairs.
    """
    if invalid is True and custom_redirector is None:
        raise ValueError("A custom redirector must not be None if invalid is True")

    file_dict = {}
    if isinstance(datasets, str):
        datasets = [datasets]

    if invalid:
        for dataset in datasets:
            file_dict[dataset] = redirect_files(
                dasgoclient_query(dataset, invalid=invalid),
                redirector=custom_redirector,
                isrucio=False,
            )

    else:
        for dataset in datasets:
            file_dict[dataset] = get_dataset_files_replicas(dataset, mode="first")[0]
            if custom_redirector:
                file_dict[dataset] = redirect_files(
                    file_dict[dataset], redirector=custom_redirector, isrucio=True
                )

    for dataset in datasets:
        print(f"Dataset {dataset} has {len(file_dict[dataset])} files\n")
        print(f"First file of dataset {dataset} is {file_dict[dataset][0]}\n")
        print(f"Last file of dataset {dataset} is {file_dict[dataset][-1]}\n")

    return file_dict


def create_fileset(file_dict):
    """Create the fileset to pass into coffea.dataset_tools.preprocess()

    Parameters
    ----------
        file_dict : dict
            The dictionary of {dataset : files} pairs.

    Returns
    -------
        fileset : dict
            The fileset to pass into coffea.dataset_tools.preprocess().
            It is a dict of the form:
            fileset = {
                "dataset": {"files": <something that uproot expects>, "metadata": {...}, ...},
                ...
            }
    """
    fileset = {}
    for dataset, files in file_dict.items():
        uproot_expected = {f: "Events" for f in files}
        fileset[dataset] = {"files": uproot_expected}

    return fileset


def get_nanoevents_file(
    names,
    *,
    toquery=False,
    redirect=False,
    custom_redirector="root://cmsxrootd.fnal.gov/",
    invalid=False,
    preprocess=False,
    preprocess_args={},
):
    """Get the `file` for NanoEventsFactory.from_root() from the given dataset names.

    Parameters
    ----------
        names : str or list of str
            The dataset names to query or a list of file paths.
        toquery : bool, optional
            Whether to query DAS for the dataset names. The default is False.
        redirect : bool, optional
            Whether to add an xrootd redirector to the files. The default is False.
        custom_redirector : str, optional
            The xrootd redirector to add to the files. The default is "root://cmsxrootd.fnal.gov/".
            Only used if redirect is True.
        invalid : bool, optional
            Whether to include invalid files. The default is False.
            Only used if toquery is True.
        preprocess : bool, optional
            Whether to preprocess the files using coffea.dataset_tools.preprocess().
            The default is False.
        preprocess_args : dict, optional
            Extra arguments to pass to coffea.dataset_tools.preprocess(). The default is {}.

    Returns
    -------
        file : a string or dict input to ``uproot.dask()``
            The filename or dict of filenames including the treepath as it would be passed directly to ``uproot.dask()``.
    """
    if redirect and custom_redirector is None:
        raise ValueError("A custom redirector must not be None if redirect is True")

    if toquery:
        if redirect:
            file_dict = get_file_dict(
                names, custom_redirector=custom_redirector, invalid=invalid
            )
        else:
            file_dict = get_file_dict(names, custom_redirector=None, invalid=invalid)

        if preprocess:
            from coffea.dataset_tools import preprocess

            print("Starting preprocessing")
            fileset = create_fileset(file_dict)
            out_available, out_updated = preprocess(fileset, **preprocess_args)
            file = {}
            for category, details in out_available.items():
                file.update(details["files"])

        else:
            file = {f: "Events" for k, files in file_dict.items() for f in files}

    else:
        if isinstance(names, str):
            names = [names]
        if redirect:
            names = redirect_files(names, redirector=custom_redirector, isrucio=False)

        file = {f: "Events" for f in names}

        if preprocess:
            from coffea.dataset_tools import preprocess

            print("Starting preprocessing")
            fileset = {"dataset": {"files": file}}
            out_available, out_updated = preprocess(fileset, **preprocess_args)
            file = out_available["dataset"]["files"]

    return file


def get_ratio_histogram(passing_probes, all_probes):
    """Get the ratio (efficiency) of the passing and all probes histograms.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        all_probes : hist.Hist
            The histogram of all probes.

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    ratio = passing_probes / all_probes
    ratio[:] = np.nan_to_num(ratio.values())
    yerr = intervals.ratio_uncertainty(
        passing_probes.values(), all_probes.values(), uncertainty_type="efficiency"
    )

    return ratio, yerr


def get_pt_and_eta_ratios(hpt_pass, hpt_all, heta_pass, heta_all):
    """Get the ratio histograms (efficiency) of pt and eta.
    NaN values are replaced with 0.

    Parameters
    ----------
        hpt_pass : hist.Hist
            The Pt histogram of the passing probes.
        hpt_all : hist.Hist
            The Pt histogram of all probes.
        heta_pass : hist.Hist
            The Eta histogram of the passing probes.
        heta_all : hist.Hist
            The Eta histogram of all probes.

    Returns
    -------
        hptratio : hist.Hist
            The Pt ratio histogram.
        hetaratio : hist.Hist
            The Eta ratio histogram.
    """
    hptratio, hptratio_yerr = get_ratio_histogram(hpt_pass, hpt_all)
    hetaratio, hetaratio_yerr = get_ratio_histogram(heta_pass, heta_all)

    return hptratio, hptratio_yerr, hetaratio, hetaratio_yerr
