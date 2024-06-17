from __future__ import annotations

import json
import re

import awkward as ak
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


def calculate_photon_SC_eta_numpy(photons, PV):
    """Calculate photon supercluster eta, following the implementation from https://github.com/bartokm/GbbMET/blob/026dac6fde5a1d449b2cfcaef037f704e34d2678/analyzer/Analyzer.h#L2487
    Before NanoAODv13, there is only the photon eta which is the SC eta corrected by the PV position.
    The SC eta is needed to correctly apply a number of corrections and systematics.
    """
    PV_x = PV.x.to_numpy()
    PV_y = PV.y.to_numpy()
    PV_z = PV.z.to_numpy()

    mask_barrel = photons.isScEtaEB
    mask_endcap = photons.isScEtaEE

    tg_theta_over_2 = np.exp(-photons.eta)
    # avoid dividion by zero
    tg_theta_over_2 = np.where(tg_theta_over_2 == 1.0, 1 - 1e-10, tg_theta_over_2)
    tg_theta = 2 * tg_theta_over_2 / (1 - tg_theta_over_2 * tg_theta_over_2)  # tg(a+b) = tg(a)+tg(b) / (1-tg(a)*tg(b))

    # calculations for EB
    R = 130.0
    angle_x0_y0 = np.zeros_like(PV_x)

    angle_x0_y0[PV_x > 0] = np.arctan(PV_y[PV_x > 0] / PV_x[PV_x > 0])
    angle_x0_y0[PV_x < 0] = np.pi + np.arctan(PV_y[PV_x < 0] / PV_x[PV_x < 0])
    angle_x0_y0[((PV_x == 0) & (PV_y >= 0))] = np.pi / 2
    angle_x0_y0[((PV_x == 0) & (PV_y < 0))] = -np.pi / 2

    alpha = angle_x0_y0 + (np.pi - photons.phi)
    sin_beta = np.sqrt(PV_x**2 + PV_y**2) / R * np.sin(alpha)
    beta = np.abs(np.arcsin(sin_beta))
    gamma = np.pi / 2 - alpha - beta
    length = np.sqrt(R**2 + PV_x**2 + PV_y**2 - 2 * R * np.sqrt(PV_x**2 + PV_y**2) * np.cos(gamma))
    z0_zSC = length / tg_theta

    tg_sctheta = ak.Array(tg_theta)
    # correct values for EB
    tg_sctheta = ak.where(mask_barrel, R / (PV_z + z0_zSC), tg_sctheta)

    # calculations for EE
    intersection_z = np.where(photons.eta > 0, 310.0, -310.0)
    base = intersection_z - PV_z
    r = base * tg_theta
    crystalX = PV_x + r * np.cos(photons.phi)
    crystalY = PV_y + r * np.sin(photons.phi)
    # correct values for EE
    tg_sctheta = ak.where(mask_endcap, np.sqrt(crystalX**2 + crystalY**2) / intersection_z, tg_sctheta)

    sctheta = np.arctan(tg_sctheta)
    sctheta = ak.where(sctheta < 0, np.pi + sctheta, sctheta)
    ScEta = -np.log(np.tan(sctheta / 2))

    return ScEta


def dask_calculate_photon_SC_eta(photons, PV):
    """Wrapper for calculate_photon_SC_eta_numpy to be used with map_partitions"""
    ak.typetracer.touch_data(photons.eta)
    ak.typetracer.touch_data(photons.phi)
    ak.typetracer.touch_data(photons.isScEtaEB)
    ak.typetracer.touch_data(photons.isScEtaEE)
    ak.typetracer.touch_data(PV.x)
    ak.typetracer.touch_data(PV.y)
    ak.typetracer.touch_data(PV.z)
    if ak.backend(photons, PV) == "typetracer":
        return ak.Array(ak.Array([[0.0]]).layout.to_typetracer(forget_length=True))
    return calculate_photon_SC_eta_numpy(photons, PV)


def calculate_photon_SC_eta(photons, PV):
    """Calculate photon supercluster eta, following the implementation from https://github.com/bartokm/GbbMET/blob/026dac6fde5a1d449b2cfcaef037f704e34d2678/analyzer/Analyzer.h#L2487
    Before NanoAODv13, there is only the photon eta which is the SC eta corrected by the PV position.
    The SC eta is needed to correctly apply a number of corrections and systematics.
    """
    PV_x = PV.x
    PV_y = PV.y
    PV_z = PV.z

    mask_barrel = photons.isScEtaEB
    mask_endcap = photons.isScEtaEE

    tg_theta_over_2 = np.exp(-photons.eta)
    # avoid dividion by zero
    tg_theta_over_2 = ak.where(tg_theta_over_2 == 1.0, 1 - 1e-10, tg_theta_over_2)
    tg_theta = 2 * tg_theta_over_2 / (1 - tg_theta_over_2 * tg_theta_over_2)  # tg(a+b) = tg(a)+tg(b) / (1-tg(a)*tg(b))

    # calculations for EB
    R = 130.0

    angle_x0_y0_positive_x = ak.where(PV_x > 0, np.arctan(PV_y / PV_x), 0)
    angle_x0_y0_negative_x = ak.where(PV_x < 0, np.pi + np.arctan(PV_y / PV_x), 0)
    angle_x0_y0_positive_y_x_0 = ak.where((PV_x == 0) & (PV_y >= 0), np.pi / 2, 0)
    angle_x0_y0_negative_y_x_0 = ak.where((PV_x == 0) & (PV_y < 0), -np.pi / 2, 0)

    angle_x0_y0 = angle_x0_y0_positive_x + angle_x0_y0_negative_x + angle_x0_y0_positive_y_x_0 + angle_x0_y0_negative_y_x_0

    alpha = angle_x0_y0 + (np.pi - photons.phi)
    sin_beta = np.sqrt(PV_x**2 + PV_y**2) / R * np.sin(alpha)
    beta = np.abs(np.arcsin(sin_beta))
    gamma = np.pi / 2 - alpha - beta
    length = np.sqrt(R**2 + PV_x**2 + PV_y**2 - 2 * R * np.sqrt(PV_x**2 + PV_y**2) * np.cos(gamma))
    z0_zSC = length / tg_theta

    tg_sctheta = tg_theta
    # correct values for EB
    tg_sctheta = ak.where(mask_barrel, R / (PV_z + z0_zSC), tg_sctheta)

    # calculations for EE
    intersection_z = ak.where(photons.eta > 0, 310.0, -310.0)
    base = intersection_z - PV_z
    r = base * tg_theta
    crystalX = PV_x + r * np.cos(photons.phi)
    crystalY = PV_y + r * np.sin(photons.phi)
    # correct values for EE
    tg_sctheta = ak.where(mask_endcap, np.sqrt(crystalX**2 + crystalY**2) / intersection_z, tg_sctheta)

    sctheta = np.arctan(tg_sctheta)
    sctheta = ak.where(sctheta < 0, np.pi + sctheta, sctheta)
    ScEta = -np.log(np.tan(sctheta / 2))

    return ScEta


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
