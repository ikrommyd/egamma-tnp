from __future__ import annotations


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
