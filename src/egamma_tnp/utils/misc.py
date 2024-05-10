import json
import re

import numba
import numpy as np


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ]
)
def delta_phi(a, b):
    """Compute difference in angle given two angles a and b

    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def delta_r(eta1, phi1, eta2, phi2):
    r"""Distance in (eta,phi) plane given two pairs of (eta,phi)

    :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.hypot(deta, dphi)


def custom_delta_r(probe, other):
    """Distance in (eta,phi) plane between probe and another object using `eta_to_use` and `phi_to_use`."""
    return delta_r(probe.eta_to_use, probe.phi_to_use, other.eta, other.phi)


def delta_r_SC(electron, other):
    """Distance in (eta,phi) plane between electron and another object using the electron's SC eta."""
    return delta_r(electron.eta + electron.deltaEtaSC, electron.phi, other.eta, other.phi)


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


def find_pt_threshold(s):
    """Find the pt threshold of a filter from the filter name.

    Parameters
    ----------
        s : str
            The filter name.
    Returns
    -------
        int
            The pt threshold.
    """
    # Extract all numbers following "Ele" in the string
    numbers = re.findall(r"Ele(\d+)", s)

    # Convert extracted numbers to integers
    numbers = [int(num) for num in numbers]

    # Return 0 if no numbers were found
    if not numbers:
        return 0
    # If 'Leg1' is in the string, return the first number
    if "Leg1" in s:
        return numbers[0]
    # Otherwise, return the second number if there are two, else return the first
    else:
        return numbers[1] if len(numbers) > 1 else numbers[0]


def replace_nans(arr):
    """Replace nans in an array with 0 before the first float and 1 after.

    Parameters
    ----------
        arr : np.array

    Returns
    -------
        np.array
            The array with nans replaced.
    """
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
