import json
import re

import numpy as np


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
    """
    # Extract all numbers following "Ele" in the string
    numbers = re.findall(r"Ele(\d+)", s)

    # Convert extracted numbers to integers
    numbers = [int(num) for num in numbers]

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
