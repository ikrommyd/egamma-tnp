from egamma_tnp.utils.rucio import dasgoclient_query, get_dataset_files_replicas


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
            print("Done preprocessing")

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
            print("Done preprocessing")

    return file
