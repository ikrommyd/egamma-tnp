import getpass
import json
import os
import re
import socket
import subprocess
from collections import defaultdict

import numpy as np
from rucio.client import Client


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


os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"


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


def get_dataset_files(
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


def get_events(custom_dataset=None):
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

    from .config import LPC, das_dataset

    if LPC:
        egamma_datasets = (
            os.popen(f"dasgoclient --query='dataset dataset={das_dataset} status=*'")
            .read()
            .splitlines()
        )

        egamma_files = {}
        for dataset in egamma_datasets:
            files = get_dataset_files(dataset)[0]
            egamma_files[dataset] = files

        for dataset in egamma_datasets:
            print(f"Dataset {dataset} has {len(egamma_files[dataset])} files\n")
            print(f"First file of dataset {dataset} is {egamma_files[dataset][0]}\n")
            print(f"Last file of dataset {dataset} is {egamma_files[dataset][-1]}\n")

        fnames = {f: "Events" for k, files in egamma_files.items() for f in files}

    elif custom_dataset:
        fnames = {f: "Events" for f in custom_dataset}

    else:
        raise Exception("No dataset specified")

    events = NanoEventsFactory.from_root(
        fnames,
        schemaclass=NanoAODSchema,
        permit_dask=True,
        chunks_per_file=1,
        metadata={"dataset": "Egamma"},
    ).events()

    return events


def get_ratio_histograms(
    hpt_pass, hpt_all, heta_pass, heta_all, habseta_pass, habseta_all
):
    hptratio = hpt_pass / hpt_all
    hptratio[:] = replace_nans(hptratio.values())

    hetaratio = heta_pass / heta_all
    hetaratio[:] = replace_nans(hetaratio.values())

    habsetaratio = habseta_pass / habseta_all
    habsetaratio[:] = replace_nans(habsetaratio.values())

    return hptratio, hetaratio, habsetaratio
