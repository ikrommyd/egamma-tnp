import getpass
import json
import os
import re
import socket
import subprocess
from collections import defaultdict

import numpy as np
from rucio.client import Client

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"


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


def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        available = True
    except Exception:
        available = False
    sock.close()
    return available


def get_proxy_path() -> str:
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


def get_rucio_client():
    try:
        nativeClient = Client(
            rucio_host="https://cms-rucio.cern.ch",
            auth_host="https://cms-rucio-auth.cern.ch",
            account=getpass.getuser(),
            creds={"client_cert": get_proxy_path(), "client_key": get_proxy_path()},
            auth_type="x509",
        )
        return nativeClient

    except Exception as e:
        print("Wrong Rucio configuration, impossible to create client")
        raise e


def get_xrootd_sites_map():
    sites_xrootd_access = defaultdict(dict)
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
    if isinstance(rules, dict):
        for rule, pfn in rules.items():
            if m := re.match(rule, path):
                grs = m.groups()
                for i in range(len(grs)):
                    pfn = pfn.replace(f"${i+1}", grs[i])
                return pfn
    else:
        return rules + "/" + path


def query_rucio(
    dataset,
    whitelist_sites=None,
    blacklist_sites=None,
    regex_sites=None,
    output="first",
):
    """
    This function queries the Rucio server to get information about the location
    of all the replicas of the files in a CMS dataset.

    The sites can be filtered in 3 different ways:
    - `whilist_sites`: list of sites to select from. If the file is not found there, raise an Exception.
    - `blacklist_sites`: list of sites to avoid. If the file has no left site, raise an Exception
    - `regex_sites`: regex expression to restrict the list of sites.

    The function can return all the possible sites for each file (`output="all"`)
    or the first site found for each file (`output="first"`, by default)
    """
    sites_xrootd_prefix = get_xrootd_sites_map()
    client = get_rucio_client()
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
                        or meta["volatile"] is True
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
                    if re.match(regex_sites, site):
                        # Check actual availability
                        meta = filedata["pfns"][rses[site][0]]
                        if (
                            meta["type"] != "DISK"
                            or meta["volatile"] is True
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
                        or meta["volatile"] is True
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
            if output == "all":
                outfiles.append(outfile)
                outsites.append(outsite)
            elif output == "first":
                outfiles.append(outfile[0])
                outsites.append(outsite[0])

    return outfiles, outsites


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


def get_das_datasets(names, *, invalid=False):
    """Get the list of datasets from DAS for the given dataset names.
    This function uses dasgoclient to query DAS as follows:
    dasgoclient --query='dataset dataset=<name> status=*'.

    Parameters
    ----------
    names : str or list of str
        The dataset names to query that can contain wildcards.
    invalid : bool, optional
        Whether to include invalid datasets. The default is False.

    Returns
    -------
    datasets: list of str
        The list of datasets.
    """
    datasets = []
    if isinstance(names, str):
        names = [names]

    if invalid:
        for query in names:
            datasets.extend(
                os.popen(f"dasgoclient --query='dataset dataset={query}'")
                .read()
                .splitlines()
            )
    else:
        for query in names:
            datasets.extend(
                os.popen(f"dasgoclient --query='dataset dataset={query} status=*'")
                .read()
                .splitlines()
            )
    return datasets


def get_files_of_das_datset(dataset, *, invalid=False):
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


def redirect_files(files, *, redirector="root://cmsxrootd.fnal.gov/"):
    """Add an xrootd redirector to a list of files

    Parameters
    ----------
    files : str or list of str
        The list of files to redirect.
    redirector : str, optional
        The xrootd redirector to add. The default is "root://cmsxrootd.fnal.gov/".
    """
    if isinstance(files, str):
        files = [files]
    return [redirector + file for file in files]


def get_file_dict(names, *, custom_redirector=None, invalid=False):
    """Get the lists of files from DAS for the given dataset names.
    The list of files is returned as a dictionary with the dataset names as keys
    and the lists of files as values.

    Parameters
    ----------
    names : str or list of str
        The dataset names to query that can contain wildcards.
    custom_redirector : str, optional
        The xrootd redirector to add to the files. The default is None.
        If None, this function will query rucio and add the redirector for the first available site.
    invalid : bool, optional
        Whether to include invalid files. The default is False.
        Only used if custom_redirector is not None.

    Returns
    -------
    file_dict: dict
        A dictionary of {dataset : files} pairs.
    """
    file_dict = {}
    datasets = get_das_datasets(names, invalid=invalid)

    if custom_redirector:
        for dataset in datasets:
            file_dict[dataset] = redirect_files(
                get_files_of_das_datset(dataset, invalid=invalid),
                redirector=custom_redirector,
            )
    else:
        for dataset in datasets:
            file_dict[dataset] = query_rucio(dataset)[0]

    for dataset in datasets:
        print(f"Dataset {dataset} has {len(file_dict[dataset])} files\n")
        print(f"First file of dataset {dataset} is {file_dict[dataset][0]}\n")
        print(f"Last file of dataset {dataset} is {file_dict[dataset][-1]}\n")

    return file_dict


def get_events(
    names,
    *,
    toquery=False,
    redirect=False,
    custom_redirector="root://cmsxrootd.fnal.gov/",
    invalid=False,
):
    """Get the NanoEvents from the given dataset names.

    Parameters
    ----------
    names : str or list of str
        The dataset names to query that can contain wildcards or a list of file paths.
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
    """
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

    if redirect and custom_redirector is None:
        raise ValueError("A custom redirector must be not be None if redirect is True")

    if toquery:
        if redirect:
            file_dict = get_file_dict(
                names, custom_redirector=custom_redirector, invalid=invalid
            )
        else:
            file_dict = get_file_dict(names, custom_redirector=None, invalid=invalid)

        fnames = {f: "Events" for k, files in file_dict.items() for f in files}

    else:
        if isinstance(names, str):
            names = [names]
        if redirect:
            names = redirect_files(names, redirector=custom_redirector)

        fnames = {f: "Events" for f in names}

    events = NanoEventsFactory.from_root(
        fnames,
        schemaclass=NanoAODSchema,
        permit_dask=True,
        chunks_per_file=1,
        metadata={"dataset": "Egamma"},
    ).events()

    return events, fnames


def get_ratio_histograms(
    hpt_all, hpt_pass, heta_all, heta_pass, habseta_all, habseta_pass
):
    hptratio = hpt_pass / hpt_all
    hptratio[:] = replace_nans(hptratio.values())

    hetaratio = heta_pass / heta_all
    hetaratio[:] = replace_nans(hetaratio.values())

    habsetaratio = habseta_pass / habseta_all
    habsetaratio[:] = replace_nans(habsetaratio.values())

    return hptratio, hetaratio, habsetaratio
