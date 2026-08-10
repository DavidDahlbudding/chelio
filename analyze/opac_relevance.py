#!/usr/bin/env python3
"""
Generate CIA opacity relevance plots for multiple sources.

This script analyzes the relevance of CIA (Collision-Induced Absorption) opacities
by comparing atmospheric opacities with and without CIA contributions across different
temperature and pressure conditions.

Usage:
    opac_relevance.py --pair "CO2-CH4" --Tmax 600
    opac_relevance.py --pair "CO2-CH4" --alpha
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap, SymLogNorm, ListedColormap, BoundaryNorm
from scipy.interpolate import interp1d, RegularGridInterpolator
from numpy.polynomial.legendre import leggauss
import sys
import os
import warnings
import h5py
import argparse
import glob

# Add HELIOS source to path
helios_source = os.environ["HELIOS_PATH"] + "/source"
if helios_source not in sys.path:
    sys.path.append(helios_source)

from species_database import species_lib
import phys_const as pc
import quantities as quant

# Set matplotlib defaults
mpl.rcParams['figure.dpi'] = 200

# ============================================================================
# Constants and Physical Data
# ============================================================================

# Physical constants (CGS units)
PLANCK_H = 6.62607015e-27  # Planck constant [erg s]
C_LIGHT = 2.99792458e10     # Speed of light [cm/s]
KB = 1.380649e-16           # Boltzmann constant [erg/K]
MMHG = 1.3328e+03           # mmHg to dyn/cm^2
BAR = 1.0e+06               # bar to dyn/cm^2

# Molecular data for species involved in CIA
names = np.array(['H2', 'He', 'N2', 'CH4', 'O2', 'CO2', 'H2O', 'NH3', 'CO'])
masses = np.array([2.016, 4.0026, 28.0134, 16.0425, 31.9988, 44.0095, 18.0153, 17.031, 28.010])  # amu

# Triple point data (from Wikipedia)
triple_Ts = np.array([13.8033, 2.1768, 63.18, 90.68, 54.36, 216.55, 273.16, 195.4, 68.1])  # K
triple_Ps = np.array([7.04e3, 5.048e3, 12.6e3, 11.7e3, 0.144e3, 517e3, 0.611657e3, 6.06e3, 15.37e3])  # Pa
triple_Ps *= 1e1  # Convert to dyn/cm^2

# Critical point data (from Wikipedia)
critical_Ts = np.array([33.20, 5.19, 126.2, 190.8, 154.33, 304.19, 647.1, 405.5, 133.16])  # K
critical_Ps = np.array([1.300e6, 0.227e6, 3.39e6, 4.64e6, 5.043e6, 7.38e6, 22.06e6, 11.28e6, 3.498e6])  # Pa
critical_Ps *= 1e1  # Convert to dyn/cm^2

# Create molecular dictionary
mol_dict = {
    name: {'mass': mass, 'triple': [Tt, Pt], 'critical': [Tc, Pc]}
    for name, mass, Tt, Pt, Tc, Pc in zip(names, masses, triple_Ts, triple_Ps, critical_Ts, critical_Ps)
}

# Default CIA sources (from run_grid_fast_cia.py)
DEFAULT_CIA_SOURCES = {
    "CH4-CH4": "CH4-CH4_2011",
    "CO2-CH4": "CO2-CH4_2024_main",
    "CO2-CO2": "CO2-CO2_2024",
    "CO2-H2": "CO2-H2_2024",
    "H2-CH4": "H2-CH4_eq_2011",
    "N2-CH4": "N2-CH4_2024_Tconst=50K",
    "N2-H2": "N2-H2_2024",
    "N2-H2O": "N2-H2O_2018",
    "N2-N2": "N2-N2_2021_Tconst=50K",
    "H2-H2": "H2-H2_2018+02",
}

# ============================================================================
# Utility Functions
# ============================================================================

def p_sat(T, species, mask=False):
    """
    Calculate saturation pressure of a given species at a given temperature.

    Args:
        T (np.ndarray): Temperature [K]
        species (str): Species name
        mask (bool): If True, set saturation pressure to NaN outside valid range

    Returns:
        np.ndarray: Saturation pressure [dyn/cm^2]
    """
    if species == 'H2':
        # NIST (21-32K)
        psat = 10.0**(3.54314 - 99.395/(T + 7.726))*BAR
        if mask:
            mask_arr = (T < 21.0) | (T > 32.0)
            psat[mask_arr] = np.nan
        return psat

    elif species == 'He':
        psat = np.empty_like(T)
        psat.fill(np.nan)  # He does not have a saturation pressure in this range
        return psat

    elif species == 'N2':
        # NIST (63-126K)
        psat = 10.0**(3.7362 - 264.651/(T - 6.788))*BAR
        if mask:
            mask_arr = (T < 63.0) | (T > 126.0)
            psat[mask_arr] = np.nan
        return psat

    elif species == 'O2':
        # NIST (54-154K)
        psat = 10.0**(3.9523 - 340.024/(T - 4.144))*BAR
        if mask:
            mask_arr = (T < 54.0) | (T > 154.0)
            psat[mask_arr] = np.nan
        return psat

    elif species == 'CH4':
        # NIST (90-190K) and GGchem
        psat = 10.0**(3.9895 - 443.028/(T-0.49))*BAR
        if mask:
            mask_arr = (T < 90.0) | (T > 190.0)
            psat[mask_arr] = np.nan
        return psat

    elif species == 'CO2':
        # Yaws' Chemical Properties Handbook (216-305K), GGchem
        C = np.array([35.0187E+00, -1.5119E+03, -1.1335E+01, 9.3383E-03, 7.7626E-10])
        psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
        psat = 10**psat * MMHG
        if mask:
            mask_arr = (T < 216.0) | (T > 305.0)
            psat[mask_arr] = np.nan
        return psat

    elif species == 'H2O':
        # Yaws' Chemical Properties Handbook (273-647K)
        C = np.array([29.8605e+00, -3.1522e+03, -7.3037e+00, 2.4247e-09, 1.8090e-06])
        psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
        psat = 10**psat * MMHG
        psat_final = psat.copy()

        # Ackerman & Marley 2001 (solid), GGchem
        TC = np.minimum(2000.0, T) - 273.15  # T[degree Celsius]
        psat = 6111.5*np.exp((23.036*TC - TC**2/333.7)/(TC + 279.82))
        psat_final[T<273.16] = psat[T<273.16]

        if mask:
            mask_arr = T > 647.0
            psat_final[mask_arr] = np.nan

        return psat_final

    elif species == 'NH3':
        # CRC Handbook of Chemistry and Physics (Weast 1971), GGchem
        psat = np.exp(10.53 - 2161.0/T - 86596.0/T**2)*BAR
        if mask:
            mask_arr = T > mol_dict['NH3']['critical'][0]  # 405.5 K
            psat[mask_arr] = np.nan
        return psat

    elif species == 'CO':
        # Yaws' Chemical Properties Handbook (68-132K), GGchem
        C = np.array([51.8145E+00, -7.8824E+02, -2.2734E+01, 5.1225E-02, 4.6603E-11])
        psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
        psat = 10**psat * MMHG
        if mask:
            mask_arr = (T < 68.0) | (T > 132.0)
            psat[mask_arr] = np.nan
        return psat

    else:
        warnings.warn(f"Saturation pressure for species {species} not found. Returning NaN.")
        return np.empty_like(T)


def get_mixing_ratio(species, T_mesh, P_mesh, x_initial=0.5):
    """
    Calculate mixing ratio grid limited by saturation vapor pressure.

    Args:
        species (str): Species name
        T_mesh (np.ndarray): Temperature grid [K]
        P_mesh (np.ndarray): Pressure grid [dyn/cm^2]
        x_initial (float): Initial mixing ratio (before saturation limit)

    Returns:
        np.ndarray: Mixing ratio grid
    """
    p_sat_grid = p_sat(T_mesh.flatten(), species, mask=False).reshape(T_mesh.shape)
    x_sat = p_sat_grid / P_mesh  # saturation mixing ratio
    return np.minimum(x_initial, x_sat)


def dB_dT(lam, T):
    """
    Calculate temperature derivative of Planck function.

    Args:
        lam (np.ndarray): Wavelength [cm]
        T (np.ndarray): Temperature [K]

    Returns:
        np.ndarray: dB/dT [erg cm^-2 s^-1 cm^-1 K^-1]
    """
    term1 = 2 * PLANCK_H**2 * C_LIGHT**3 / (lam**6 * KB * T**2)
    with np.errstate(over='ignore'):
        exp_term = np.exp(PLANCK_H * C_LIGHT / (lam * KB * T))
        term2 = exp_term / (exp_term - 1)**2
    return term1 * term2


def expand_wave(interwave, gauss_y):
    """
    Expand wavelength grid with Gaussian quadrature points.

    Args:
        interwave (np.ndarray): Interface wavelengths
        gauss_y (np.ndarray): Gaussian y-points

    Returns:
        np.ndarray: Expanded wavelength grid
    """
    x = (gauss_y[None, :] - 0.5) * 2
    dwave = (interwave[1:] - interwave[:-1])[:, None]
    mean_wave = (interwave[1:] + interwave[:-1])[:, None] / 2
    x = 0.5 * dwave * x + mean_wave
    return x


def scale_height(T, mu, M_P=5.972e27, R_P=6.371e8):
    """
    Calculate atmospheric scale height.

    Args:
        T (np.ndarray): Temperature [K]
        mu (np.ndarray): Mean molecular mass [g]
        M_P (float): Planet mass [g]
        R_P (float): Planet radius [cm]

    Returns:
        np.ndarray: Scale height [cm]
    """
    g = pc.G * M_P / R_P**2
    return KB * T / (mu * g)


def read_opac_file(quant_obj, name, type="premixed", read_grid_parameters=False):
    """
    Read opacity table file for an individual species.

    Args:
        quant_obj: Quantities object to store data
        name (str): Path to HDF5 file
        type (str): File type ("premixed" or "other")
        read_grid_parameters (bool): Whether to read grid parameters

    Returns:
        np.ndarray: Opacity k-points
    """
    with h5py.File(name, "r") as opac_file:
        print(f"\nReading opacity file: {name}")

        try:
            opac_k = [k for k in opac_file["kpoints"][:]]
        except KeyError:
            opac_k = [k for k in opac_file["opacities"][:]]

        if type == "premixed":
            # Rayleigh scattering cross-sections
            quant_obj.opac_scat_cross = [c for c in opac_file["weighted Rayleigh cross-sections"][:]]
            # Pre-tabulated mean molecular mass values
            quant_obj.opac_meanmass = [m * pc.AMU for m in opac_file["meanmolmass"][:]]

        if type == "premixed" or read_grid_parameters is True:
            # Wavelength grid
            try:
                quant_obj.opac_wave = [x for x in opac_file["center wavelengths"][:]]
            except KeyError:
                quant_obj.opac_wave = [x for x in opac_file["wavelengths"][:]]
            quant_obj.nbin = np.int32(len(quant_obj.opac_wave))

            # Gaussian y-points
            try:
                quant_obj.gauss_y = [y for y in opac_file["ypoints"][:]]
            except KeyError:
                quant_obj.gauss_y = [0]
            quant_obj.ny = np.int32(len(quant_obj.gauss_y))

            # Interface positions of the wavelength bins
            try:
                quant_obj.opac_interwave = [i for i in opac_file["interface wavelengths"][:]]
            except KeyError:
                # Derive interface values from center wavelengths
                quant_obj.opac_interwave = []
                quant_obj.opac_interwave.append(quant_obj.opac_wave[0] - (quant_obj.opac_wave[1] - quant_obj.opac_wave[0]) / 2)
                for x in range(len(quant_obj.opac_wave) - 1):
                    quant_obj.opac_interwave.append((quant_obj.opac_wave[x + 1] + quant_obj.opac_wave[x]) / 2)
                quant_obj.opac_interwave.append(quant_obj.opac_wave[-1] + (quant_obj.opac_wave[-1] - quant_obj.opac_wave[-2]) / 2)

            # Widths of the wavelength bins
            try:
                quant_obj.opac_deltawave = [w for w in opac_file["wavelength width of bins"][:]]
            except KeyError:
                quant_obj.opac_deltawave = []
                for x in range(len(quant_obj.opac_interwave) - 1):
                    quant_obj.opac_deltawave.append(quant_obj.opac_interwave[x + 1] - quant_obj.opac_interwave[x])

            # Temperature grid
            quant_obj.ktemp = [t for t in opac_file["temperatures"][:]]
            quant_obj.ntemp = np.int32(len(quant_obj.ktemp))

            # Pressure grid
            quant_obj.kpress = [p for p in opac_file["pressures"][:]]
            quant_obj.npress = np.int32(len(quant_obj.kpress))

    return opac_k


# ============================================================================
# File Discovery and Loading
# ============================================================================

def find_cia_sources_for_pair(pair, hitran_dir):
    """
    Find all CIA opacity source files for a given pair.

    Args:
        pair (str): CIA pair name (e.g., "CO2-CH4")
        hitran_dir (str): Path to hitran_cia directory

    Returns:
        list: List of (source_name, filepath) tuples
    """
    pattern = f"{pair}_*.h5"
    files = glob.glob(os.path.join(hitran_dir, pattern))

    # Extract source names from filenames
    sources = []
    for filepath in files:
        filename = os.path.basename(filepath)
        source_name = filename.replace(".h5", "")
        sources.append((source_name, filepath))

    # Sort by source name
    sources.sort(key=lambda x: x[0])

    return sources


def load_base_opacity(cia_pair, opacity_folder):
    """
    Load the "base" opacity: individual absorbers + non-target CIA pairs.
    This is the part that stays constant across all target CIA sources,
    and is loaded only once for efficiency.

    Args:
        cia_pair (str): CIA pair name (e.g., "CO2-CH4")
        opacity_folder (str): Path to r50_kdistr folder

    Returns:
        tuple: (base_quant with flat opac_k, mu_2d, mr_1_tiled, mr_2_tiled, mu_tiled)
    """
    abs_species = np.array(['CH4', 'CO2', 'H2O'])
    cia_pair_split = cia_pair.split('-')
    target_pair_name = cia_pair.replace('-', '')

    # Collect files: individual absorbers + all relevant CIA pairs
    files = []
    for species in cia_pair_split:
        if species in abs_species:
            files.append(f"{species}_opac_ip_kdistr.h5")
    files = np.unique(files).tolist()

    cia_pairs_all = np.array(["CH4-CH4", "CO2-CH4", "CO2-CO2", "CO2-H2", "H2-CH4",
                              "H2-He", "N2-CH4", "N2-H2", "N2-H2O", "N2-N2", "H2-H2"])
    for compair in cia_pairs_all:
        compair_split = compair.split('-')
        if (compair_split[0] in cia_pair_split) and (compair_split[1] in cia_pair_split):
            files.append(f"CIA_{compair_split[0]}{compair_split[1]}_opac_ip_kdistr.h5")

    len_tail = len("_opac_ip_kdistr.h5")

    # Initialize quantity object and read grid parameters
    q = quant.Store()
    q.opac_k = np.array(read_opac_file(q, opacity_folder + 'CO2_opac_ip_kdistr.h5',
                                       type='other', read_grid_parameters=True))
    q.opac_k[:] = 0.0

    T_grid = np.array(q.ktemp)
    P_grid = np.array(q.kpress)
    T_mesh, P_mesh = np.meshgrid(T_grid, P_grid, indexing='ij')

    # Calculate mixing ratios
    if cia_pair_split[0] == cia_pair_split[1]:
        mr_1 = get_mixing_ratio(cia_pair_split[0], T_mesh, P_mesh, x_initial=1.0)
        mr_2 = mr_1.copy()
        mu = species_lib[cia_pair_split[0]].weight * pc.AMU * mr_1
        mu /= mr_1
    else:
        mr_1 = get_mixing_ratio(cia_pair_split[0], T_mesh, P_mesh, x_initial=0.5)
        mr_2 = get_mixing_ratio(cia_pair_split[1], T_mesh, P_mesh, x_initial=1-mr_1)
        mr_1 = get_mixing_ratio(cia_pair_split[0], T_mesh, P_mesh, x_initial=1-mr_2)
        mu = (species_lib[cia_pair_split[0]].weight * pc.AMU * mr_1 +
              species_lib[cia_pair_split[1]].weight * pc.AMU * mr_2)
        mu /= (mr_1 + mr_2)

    mr_1_tiled = np.tile(mr_1[:, :, None, None], (1, 1, q.nbin, q.ny)).flatten()
    mr_2_tiled = np.tile(mr_2[:, :, None, None], (1, 1, q.nbin, q.ny)).flatten()
    mu_tiled = np.tile(mu[:, :, None, None], (1, 1, q.nbin, q.ny)).flatten()

    # Load and combine base opacities (everything EXCEPT the target CIA pair)
    for file in files:
        filepath = opacity_folder + file
        if not os.path.exists(filepath):
            continue

        # Skip the target CIA pair entirely
        is_target_cia = file[:3] == 'CIA' and file.split('_')[1] == target_pair_name
        if is_target_cia:
            print(f'  Skipping target CIA pair: {file}')
            continue

        new_opac_k = np.array(read_opac_file(q, filepath, type='other', read_grid_parameters=False))
        factor = species_lib[file[:-len_tail]].weight * pc.AMU / mu_tiled

        if file[:3] == 'CIA':
            print(f'  Other CIA pair: {file}')
            if file.split('_')[1] == cia_pair_split[0]+cia_pair_split[0]:
                mr_factor = mr_1_tiled**2
            elif file.split('_')[1] == cia_pair_split[1]+cia_pair_split[1]:
                mr_factor = mr_2_tiled**2
            else:
                raise ValueError(f"Wrong CIA pair: {file.split('_')[1]}")
            q.opac_k += new_opac_k * factor * mr_factor
        else:
            print(f'  Individual absorber: {file}')
            if file.split('_')[0] == cia_pair_split[0]:
                mr_factor = mr_1_tiled
            elif file.split('_')[0] == cia_pair_split[1]:
                mr_factor = mr_2_tiled
            else:
                raise ValueError(f"Wrong species: {file.split('_')[0]}")
            q.opac_k += new_opac_k * factor * mr_factor

    return q, mu, mr_1_tiled, mr_2_tiled, mu_tiled


def build_quant_with_cia(base_quant, cia_source_path, cia_pair, mr_1_tiled, mr_2_tiled, mu_tiled):
    """
    Build a complete opacity quant by adding a target CIA source on top of the base.

    Uses the same weighting as the notebook:
        factor = species_lib["CIA_XY"].weight * pc.AMU / mu

    Args:
        base_quant: Quant object with flat base opac_k (will NOT be mutated)
        cia_source_path (str): Path to CIA source h5 file from hitran_cia
        cia_pair (str): CIA pair name (e.g., "CO2-CH4")
        mr_1_tiled (np.ndarray): Mixing ratio of species 1 (flat)
        mr_2_tiled (np.ndarray): Mixing ratio of species 2 (flat)
        mu_tiled (np.ndarray): Mean molecular mass (flat)

    Returns:
        quant object with combined opacity (reshaped, with gauss_weight and opac_band_lay)
    """
    cia_pair_split = cia_pair.split('-')
    target_pair_name = cia_pair.replace('-', '')

    # Copy grid parameters from base
    q = quant.Store()
    q.ntemp = base_quant.ntemp
    q.npress = base_quant.npress
    q.nbin = base_quant.nbin
    q.ny = base_quant.ny
    q.ktemp = base_quant.ktemp
    q.kpress = base_quant.kpress
    q.opac_wave = base_quant.opac_wave
    q.opac_interwave = base_quant.opac_interwave
    q.opac_deltawave = base_quant.opac_deltawave
    q.gauss_y = base_quant.gauss_y

    # Start from a copy of the base opacity
    opac_k = base_quant.opac_k.copy()

    # Load and add target CIA from hitran_cia
    print(f'  Loading target CIA from: {cia_source_path}')
    new_opac_k = np.array(read_opac_file(q, cia_source_path, type='other', read_grid_parameters=False))

    # Weight consistently with the notebook: species_lib["CIA_XY"].weight * AMU / mu
    factor = species_lib[f"CIA_{target_pair_name}"].weight * pc.AMU / mu_tiled

    if cia_pair_split[0] == cia_pair_split[1]:
        mr_factor = mr_1_tiled**2
    else:
        mr_factor = mr_1_tiled * mr_2_tiled
    opac_k += new_opac_k * factor * mr_factor

    # Handle very small opacities
    q.opac_k = np.maximum(opac_k, 1e-15)

    # Reshape and compute band-integrated opacity
    q.opac_k = q.opac_k.reshape(q.ntemp, q.npress, q.nbin, q.ny)
    q.gauss_weight = leggauss(q.ny)[1]
    q.opac_band_lay = np.sum(0.5 * q.gauss_weight * q.opac_k, axis=-1)

    return q


def finalize_base_quant(base_quant):
    """
    Finalize the base quant (without target CIA) for Rosseland mean calculation.

    Args:
        base_quant: Quant object with flat opac_k

    Returns:
        New quant object reshaped, with gauss_weight and opac_band_lay set
    """
    q = quant.Store()
    q.ntemp = base_quant.ntemp
    q.npress = base_quant.npress
    q.nbin = base_quant.nbin
    q.ny = base_quant.ny
    q.ktemp = base_quant.ktemp
    q.kpress = base_quant.kpress
    q.opac_wave = base_quant.opac_wave
    q.opac_interwave = base_quant.opac_interwave
    q.opac_deltawave = base_quant.opac_deltawave
    q.gauss_y = base_quant.gauss_y
    q.opac_k = np.maximum(base_quant.opac_k.copy(), 1e-15)
    q.opac_k = q.opac_k.reshape(q.ntemp, q.npress, q.nbin, q.ny)
    q.gauss_weight = leggauss(q.ny)[1]
    q.opac_band_lay = np.sum(0.5 * q.gauss_weight * q.opac_k, axis=-1)
    return q


# ============================================================================
# Rosseland Mean and Optical Depth Calculations
# ============================================================================

def calculate_rosseland_mean(q):
    """
    Calculate Rosseland mean opacity.

    Args:
        q: Quant object with opacity data

    Returns:
        np.ndarray: Rosseland mean opacity (ntemp, npress)
    """
    q.opac_interwave = np.array(q.opac_interwave)
    q.opac_deltawave = np.array(q.opac_deltawave)
    q.gauss_y = np.array(q.gauss_y)
    q.ktemp = np.array(q.ktemp)
    q.kpress = np.array(q.kpress)

    # Expand wavelength grid with Gaussian points
    expanded_wave = expand_wave(q.opac_interwave, q.gauss_y)

    # Calculate dB/dT
    dBdT = dB_dT(expanded_wave[None, None, :], q.ktemp[:, None, None, None])
    dBdT[np.isnan(dBdT)] = np.finfo(float).eps

    # Integrate over wavelength
    integrated_dB_dT = np.sum(0.5 * q.opac_deltawave[None, None, :, None] *
                              q.gauss_weight[None, None, None, :] * dBdT, axis=-1)
    num_ross = np.sum(integrated_dB_dT, axis=-1)

    # Calculate Rosseland mean
    denom_ross = np.sum(0.5 * q.opac_deltawave[None, None, :, None] *
                        q.gauss_weight[None, None, None, :] * dBdT / q.opac_k,
                        axis=(-1, -2))
    ross_opac = num_ross / denom_ross

    return ross_opac


def calculate_optical_depth(ross_opac, mu, T_grid, P_grid):
    """
    Calculate optical depth = kappa * rho * H.

    Args:
        ross_opac (np.ndarray): Rosseland mean opacity (ntemp, npress)
        mu (np.ndarray): Mean molecular mass [g] (ntemp, npress)
        T_grid (np.ndarray): Temperature grid [K]
        P_grid (np.ndarray): Pressure grid [dyn/cm^2]

    Returns:
        np.ndarray: Optical depth (ntemp, npress)
    """
    # Calculate density
    density = mu * P_grid[None, :] / (KB * T_grid[:, None])

    # Calculate scale height
    h_scale = scale_height(T_grid[:, None], mu)

    # Calculate optical depth
    tau = ross_opac * density * h_scale

    return tau


def find_rcb_pressure(ross_opac, T_grid, P_grid, mu_grid, M_P=5.972e27, R_P=6.371e8):
    """
    Find pressure where optical depth tau ≈ 1 for each temperature.

    This marks the approximate radiative-convective boundary (RCB).

    Args:
        ross_opac (np.ndarray): Rosseland mean opacity (ntemp, npress) [cm²/g]
        T_grid (np.ndarray): Temperature grid [K]
        P_grid (np.ndarray): Pressure grid [dyn/cm²]
        mu_grid (np.ndarray): Mean molecular mass grid (ntemp, npress) [g]
        M_P (float): Planet mass [g] (default: Earth)
        R_P (float): Planet radius [cm] (default: Earth)

    Returns:
        np.ndarray: P_rcb array (ntemp,) with NaN for unsolvable T values [dyn/cm²]
    """
    # Calculate surface gravity
    g = pc.G * M_P / R_P**2  # [cm/s²]

    # Calculate optical depth: tau = kappa * P / g
    # This simplified form comes from tau = kappa * rho * H
    # where rho = mu*P/(k_B*T) and H = k_B*T/(mu*g)
    # The mu and k_B*T factors cancel, giving tau = kappa * P / g
    tau = ross_opac * P_grid[None, :] / g  # (ntemp, npress)

    # For each temperature, find pressure where tau is closest to 1.0
    P_rcb = np.full(len(T_grid), np.nan)

    for i_T in range(len(T_grid)):
        tau_at_T = tau[i_T, :]

        # Skip if all tau values are invalid
        if np.all(np.isnan(tau_at_T)) or np.all(tau_at_T == 0):
            continue

        # Check if tau=1 is bracketed by the pressure grid
        if np.nanmin(tau_at_T) > 1.0 or np.nanmax(tau_at_T) < 1.0:
            # tau=1 is outside the grid range
            continue

        # Find the two pressures that bracket tau=1
        # tau typically increases with pressure, but may not be monotonic
        # Find where tau crosses 1.0
        diff_from_one = np.abs(tau_at_T - 1.0)
        i_closest = np.nanargmin(diff_from_one)

        # Use linear interpolation if we can bracket the solution
        if i_closest > 0 and i_closest < len(P_grid) - 1:
            # Check both neighbors
            tau_prev = tau_at_T[i_closest - 1]
            tau_curr = tau_at_T[i_closest]
            tau_next = tau_at_T[i_closest + 1]

            # Interpolate with the neighbor that better brackets tau=1
            if (tau_prev - 1.0) * (tau_curr - 1.0) < 0:
                # Bracketed between prev and curr
                frac = (1.0 - tau_prev) / (tau_curr - tau_prev)
                P_rcb[i_T] = P_grid[i_closest - 1] * (1 - frac) + P_grid[i_closest] * frac
            elif (tau_curr - 1.0) * (tau_next - 1.0) < 0:
                # Bracketed between curr and next
                frac = (1.0 - tau_curr) / (tau_next - tau_curr)
                P_rcb[i_T] = P_grid[i_closest] * (1 - frac) + P_grid[i_closest + 1] * frac
            else:
                # Not bracketed, just use closest
                P_rcb[i_T] = P_grid[i_closest]
        else:
            # Edge case: use closest point
            P_rcb[i_T] = P_grid[i_closest]

    return P_rcb


def interpolate_fine_T_grid(ross_opac, mu_grid, T_grid, P_grid, fine_Tstep, Tmax):
    """
    Interpolate Rosseland opacity and mean molecular mass to finer T grid.

    Opacity is interpolated logarithmically (log10) to avoid artifacts.

    Args:
        ross_opac (np.ndarray): Rosseland mean opacity (ntemp, npress) [cm²/g]
        mu_grid (np.ndarray): Mean molecular mass grid (ntemp, npress) [g]
        T_grid (np.ndarray): Temperature grid [K]
        P_grid (np.ndarray): Pressure grid [dyn/cm²]
        fine_Tstep (int): Target temperature step size [K]
        Tmax (float): Maximum temperature [K]

    Returns:
        tuple: (ross_opac_fine, mu_fine, T_fine)
            - ross_opac_fine (np.ndarray): Interpolated Rosseland opacity (ntemp_fine, npress)
            - mu_fine (np.ndarray): Interpolated mean molecular mass (ntemp_fine, npress)
            - T_fine (np.ndarray): Fine temperature grid [K]
    """
    # Create fine temperature grid
    T_fine = np.arange(T_grid[0], min(T_grid[-1], Tmax) + fine_Tstep, fine_Tstep)

    # Ensure we don't go beyond original grid
    T_fine = T_fine[T_fine <= T_grid[-1]]

    # Check if fine grid is actually finer than native grid
    native_Tstep = np.min(np.diff(T_grid))
    if fine_Tstep >= native_Tstep:
        warnings.warn(f"Requested fine_Tstep ({fine_Tstep}K) is not finer than "
                     f"native grid spacing ({native_Tstep}K). Using native grid.")
        return ross_opac, mu_grid, T_grid

    print(f"  Interpolating to fine T grid: {len(T_grid)} -> {len(T_fine)} points "
          f"(ΔT: {native_Tstep}K -> {fine_Tstep}K)")

    # Use RegularGridInterpolator for 2D interpolation
    # Interpolate log10(Rosseland opacity) for smoother results across orders of magnitude
    log_ross_opac = np.log10(ross_opac)
    interp_log_ross = RegularGridInterpolator(
        (T_grid, P_grid), log_ross_opac,
        method='linear', bounds_error=False, fill_value=None
    )

    # Interpolate mean molecular mass (linear is fine for this)
    interp_mu = RegularGridInterpolator(
        (T_grid, P_grid), mu_grid,
        method='linear', bounds_error=False, fill_value=None
    )

    # Create meshgrid for evaluation
    T_fine_mesh, P_fine_mesh = np.meshgrid(T_fine, P_grid, indexing='ij')
    points = np.column_stack([T_fine_mesh.ravel(), P_fine_mesh.ravel()])

    # Evaluate interpolators
    log_ross_opac_fine = interp_log_ross(points).reshape(len(T_fine), len(P_grid))
    # Convert back from log space
    ross_opac_fine = 10**log_ross_opac_fine
    mu_fine = interp_mu(points).reshape(len(T_fine), len(P_grid))

    return ross_opac_fine, mu_fine, T_fine


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_relevance_map(quant_with, quant_without, cia_pair, source_name, Tmax,
                       output_dir, use_alpha, mu_grid, T_grid, P_grid, fine_Tstep=None):
    """
    Create and save relevance plot showing log10(kappa_with / kappa_without).

    Args:
        quant_with: Quant object with target CIA pair
        quant_without: Quant object without target CIA pair
        cia_pair (str): CIA pair name
        source_name (str): Source name (e.g., "CO2-CH4_2024_main")
        Tmax (int): Maximum temperature for plot
        output_dir (str): Output directory path
        use_alpha (bool): Whether to use alpha transparency
        mu_grid (np.ndarray): Mean molecular mass grid
        T_grid (np.ndarray): Temperature grid
        P_grid (np.ndarray): Pressure grid
        fine_Tstep (int): Optional fine temperature step size [K]
    """
    # Calculate Rosseland means
    print(f"Calculating Rosseland mean opacities...")
    ross_with = calculate_rosseland_mean(quant_with)
    ross_without = calculate_rosseland_mean(quant_without)

    # Apply fine temperature interpolation if requested
    if fine_Tstep is not None:
        # Save original T_grid before any interpolation
        T_grid_original = T_grid.copy()

        ross_with, mu_grid_with, T_grid = interpolate_fine_T_grid(
            ross_with, mu_grid, T_grid_original, P_grid, fine_Tstep, Tmax
        )
        ross_without, mu_grid_without, T_grid_fine = interpolate_fine_T_grid(
            ross_without, mu_grid, T_grid_original, P_grid, fine_Tstep, Tmax
        )
        # Use mu_grid from 'with' case for consistency
        mu_grid = mu_grid_with
    else:
        T_grid_original = T_grid  # No interpolation, keep reference to original

    # Limit to Tmax
    i_Tmax = np.where(T_grid > Tmax)[0]
    if len(i_Tmax) > 0:
        i_Tmax = i_Tmax[0]
    else:
        i_Tmax = len(T_grid)

    T_plot = T_grid[:i_Tmax]
    logP_plot = np.log10(P_grid * 1e-6)  # Convert to bar

    # Calculate relevance ratio
    rosseland_plot = ross_with[:i_Tmax, :] / ross_without[:i_Tmax, :]
    rosseland_plot = np.log10(rosseland_plot)

    # Calculate alpha transparency if requested
    if use_alpha:
        tau = calculate_optical_depth(ross_with[:i_Tmax, :], mu_grid[:i_Tmax, :],
                                       T_plot, P_grid)
        alpha = np.minimum(tau, 1.0)
        alpha[np.isnan(alpha)] = 0
        # Only show where tau > 0.51
        alpha[tau <= 0.51] = 0
        vmax = rosseland_plot[alpha > 0.51].max() if (alpha > 0.51).any() else rosseland_plot.max()
    else:
        alpha = np.ones_like(rosseland_plot)
        vmax = None

    # Calculate RCB if alpha is off (for plotting tau=1 contour)
    P_rcb = None
    if not use_alpha:
        print(f"Calculating RCB (tau=1 contour)...")
        P_rcb = find_rcb_pressure(ross_with[:i_Tmax, :], T_plot, P_grid, mu_grid[:i_Tmax, :])

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(rosseland_plot.T[::-1, :], alpha=alpha.T[::-1, :],
                   origin='lower', aspect='auto', vmax=vmax, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(\kappa_{\text{with CIA}} / \kappa_{\text{without CIA}})$')

    # Add saturation vapor pressure overlays
    cia_pair_split = cia_pair.split('-')
    ax2 = ax.twinx()
    ax.set_ylim(ax.get_ylim())

    Ts = np.linspace(25, Tmax + 25, 100)
    T_plots = (Ts - Ts.min()) / (Ts.max() - Ts.min()) * i_Tmax - 0.5
    hatches = ['..', '//']

    for i_species, species in enumerate(cia_pair_split):
        p_sats = np.log10(p_sat(Ts, species, mask=False) * 1e-6)
        ax2.fill_between(T_plots, p_sats, 4, label=species, hatch=hatches[i_species],
                        color='none', edgecolor='black')

        # Add phase points
        for phase_point, phase_symbol in zip([mol_dict[species]['triple'],
                                              mol_dict[species]['critical']], ['+', 'x']):
            T_phase_point = (phase_point[0] - Ts.min()) / (Ts.max() - Ts.min()) * i_Tmax - 0.5
            ax2.scatter(T_phase_point, np.log10(phase_point[1] * 1e-6),
                       marker=phase_symbol, color='black', s=100, linewidths=2)

        # Don't duplicate for same species pairs
        if cia_pair_split[0] == cia_pair_split[1]:
            break

    ax2.set_xlim(-0.5, i_Tmax - 0.5)
    ax2.set_ylim(3 + 0.5/3, -(6 + 0.5/3))
    ax2.set_yticks([])

    # Add RCB overlay (tau=1 contour) if calculated
    if P_rcb is not None:
        # Convert P_rcb to plot coordinates
        # X: T index (already 0 to i_Tmax-1)
        # Y: log10(P/bar) reversed
        T_indices = np.arange(len(T_plot))
        P_rcb_log = np.log10(P_rcb * 1e-6)  # Convert to log10(bar)

        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(P_rcb_log)
        if np.any(valid_mask):
            ax2.plot(T_indices[valid_mask], P_rcb_log[valid_mask],
                    color='red', linewidth=2.5, linestyle='--',
                    label='τ=1 (RCB)', zorder=10)

    ax2.legend()

    # Set ticks and labels
    # For fine T grid, use sparser ticks based on original grid
    if fine_Tstep is not None:
        # Find indices in T_plot that correspond to original T grid values
        original_T_in_range = T_grid_original[T_grid_original <= Tmax]
        tick_indices = []
        tick_labels = []
        for i, T_orig in enumerate(original_T_in_range):
            # Find closest index in T_plot
            idx = np.argmin(np.abs(T_plot - T_orig))
            if idx not in tick_indices:  # Avoid duplicates
                tick_indices.append(idx)
                tick_labels.append(f"{T_orig:.0f}" if (i+1) % 2 == 0 else " ")
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45)
    else:
        # Original behavior for non-fine grid
        ax.set_xticks(np.arange(len(T_plot)))
        ax.set_xticklabels([f"{v:.0f}" if (i+1) % 2 == 0 else " "
                            for i, v in enumerate(T_plot)], rotation=45)
    ax.set_xlabel('T [K]')

    ax.set_yticks(np.arange(len(logP_plot)))
    ax.set_yticklabels([f"{v:.1g}" if f"{v:.1f}"[-1] == "0" else " "
                        for v in logP_plot[::-1]])
    ax.set_ylabel('log(P) [bar]')

    plt.tight_layout()

    # Save figure
    suffix = "_alpha" if use_alpha else ""
    filename = f"{source_name}_rosseland_mean_ratio{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_difference_map(quant_source, quant_default, cia_pair, source_name, default_name,
                       Tmax, output_dir, use_alpha, mu_source, T_grid, P_grid, fine_Tstep=None):
    """
    Create and save difference plot showing log10(kappa_source / kappa_default).

    Args:
        quant_source: Quant object for non-default source
        quant_default: Quant object for default source
        cia_pair (str): CIA pair name
        source_name (str): Non-default source name
        default_name (str): Default source name
        Tmax (int): Maximum temperature for plot
        output_dir (str): Output directory path
        use_alpha (bool): Whether to use alpha transparency
        mu_source (np.ndarray): Mean molecular mass grid for source
        T_grid (np.ndarray): Temperature grid
        P_grid (np.ndarray): Pressure grid
        fine_Tstep (int): Optional fine temperature step size [K]
    """
    # Calculate Rosseland means
    print(f"Calculating Rosseland mean opacities for difference plot...")
    ross_source = calculate_rosseland_mean(quant_source)
    ross_default = calculate_rosseland_mean(quant_default)

    # Apply fine temperature interpolation if requested
    if fine_Tstep is not None:
        # Save original T_grid and mu_source before any interpolation
        T_grid_original = T_grid.copy()
        mu_source_original = mu_source.copy()

        ross_source, mu_source, T_grid = interpolate_fine_T_grid(
            ross_source, mu_source_original, T_grid_original, P_grid, fine_Tstep, Tmax
        )
        ross_default, _, T_grid_fine = interpolate_fine_T_grid(
            ross_default, mu_source_original, T_grid_original, P_grid, fine_Tstep, Tmax
        )
    else:
        T_grid_original = T_grid  # No interpolation, keep reference to original

    # Limit to Tmax
    i_Tmax = np.where(T_grid > Tmax)[0]
    if len(i_Tmax) > 0:
        i_Tmax = i_Tmax[0]
    else:
        i_Tmax = len(T_grid)

    T_plot = T_grid[:i_Tmax]
    logP_plot = np.log10(P_grid * 1e-6)

    # Calculate difference ratio
    diff_plot = ross_source[:i_Tmax, :] / ross_default[:i_Tmax, :]
    diff_plot = np.log10(diff_plot)

    # Calculate alpha transparency if requested
    if use_alpha:
        tau = calculate_optical_depth(ross_source[:i_Tmax, :], mu_source[:i_Tmax, :],
                                      T_plot, P_grid)
        alpha = np.minimum(tau, 1.0)
        alpha[np.isnan(alpha)] = 0
    else:
        alpha = np.ones_like(diff_plot)

    # Calculate RCB if alpha is off (for plotting tau=1 contour)
    P_rcb = None
    if not use_alpha:
        print(f"Calculating RCB (tau=1 contour)...")
        P_rcb = find_rcb_pressure(ross_source[:i_Tmax, :], T_plot, P_grid, mu_source[:i_Tmax, :])

    # Create figure with symmetric colormap
    fig, ax = plt.subplots(figsize=(4, 6))

    # Use symmetric colormap centered at zero
    vmax = np.abs(diff_plot).max()
    im = ax.imshow(diff_plot.T[::-1, :], alpha=alpha.T[::-1, :],
                   origin='lower', aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(\kappa_{\text{source}} / \kappa_{\text{default}})$')

    # Set ticks and labels
    # For fine T grid, use sparser ticks based on original grid
    if fine_Tstep is not None:
        # Find indices in T_plot that correspond to original T grid values
        original_T_in_range = T_grid_original[T_grid_original <= Tmax]
        tick_indices = []
        tick_labels = []
        for i, T_orig in enumerate(original_T_in_range):
            # Find closest index in T_plot
            idx = np.argmin(np.abs(T_plot - T_orig))
            if idx not in tick_indices:  # Avoid duplicates
                tick_indices.append(idx)
                tick_labels.append(f"{T_orig:.0f}" if (i+1) % 2 == 0 else " ")
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45)
    else:
        # Original behavior for non-fine grid
        ax.set_xticks(np.arange(len(T_plot)))
        ax.set_xticklabels([f"{v:.0f}" if (i+1) % 2 == 0 else " "
                            for i, v in enumerate(T_plot)], rotation=45)
    ax.set_xlabel('T [K]')

    ax.set_yticks(np.arange(len(logP_plot)))
    ax.set_yticklabels([f"{v:.1g}" if f"{v:.1f}"[-1] == "0" else " "
                        for v in logP_plot[::-1]])
    ax.set_ylabel('log(P) [bar]')

    # Add title showing comparison
    source_label = source_name.replace(f"{cia_pair}_", "")
    default_label = default_name.replace(f"{cia_pair}_", "")
    ax.set_title(f"{source_label} vs {default_label}", loc='left', fontsize=10)

    # Add RCB overlay (tau=1 contour) if calculated
    if P_rcb is not None:
        # Convert P_rcb to plot coordinates
        T_indices = np.arange(len(T_plot))
        P_rcb_log = np.log10(P_rcb * 1e-6)  # Convert to log10(bar)

        # Map to reversed y-axis (imshow has origin='lower', so reversed)
        # Y-axis goes from 0 (top, high log(P)) to len(P_grid)-1 (bottom, low log(P))
        # We need to convert log10(P) to pixel index
        logP_plot_full = np.log10(P_grid * 1e-6)

        # Interpolate P_rcb_log to pixel indices
        valid_mask = ~np.isnan(P_rcb_log)
        if np.any(valid_mask):
            # P_rcb_log is in log10(bar), need to convert to pixel index
            # The y-axis is reversed in imshow, going from high to low pressure
            y_indices = np.interp(P_rcb_log[valid_mask], logP_plot_full[::-1],
                                 np.arange(len(logP_plot_full))[::-1])

            ax.plot(T_indices[valid_mask], y_indices,
                   color='red', linewidth=2.5, linestyle='--',
                   label='τ=1 (RCB)', zorder=10)
            ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save figure
    suffix = "_alpha" if use_alpha else ""
    filename = f"{source_name}_vs_{default_label}_difference{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"Saved: {filepath}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Generate CIA opacity relevance plots for multiple sources.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pair "CO2-CH4" --Tmax 600
  %(prog)s --pair "CO2-CH4" --alpha
  %(prog)s --pair "N2-H2" --Tmax 400 --alpha
  %(prog)s --pair "CO2-CH4" --fine-Tstep 10
  %(prog)s --pair "CO2-CH4" --fine-Tstep 10 --alpha
        """
    )
    parser.add_argument('--pair', type=str, required=True,
                       help='CIA pair name (e.g., "CO2-CH4", "H2-H2")')
    parser.add_argument('--Tmax', type=int, default=600,
                       help='Maximum temperature for plots [K] (default: 600)')
    parser.add_argument('--alpha', action='store_true', default=False,
                       help='Enable alpha transparency on all plots based on optical depth')
    parser.add_argument('--fine-Tstep', type=int, default=None,
                       help='Temperature step size for interpolation [K] (default: use native grid)')

    args = parser.parse_args()

    # Set up paths
    helios_path = os.environ.get("HELIOS_PATH")
    if helios_path is None:
        print("ERROR: HELIOS_PATH environment variable not set!")
        sys.exit(1)

    hitran_dir = os.path.join(helios_path, "input", "opacity", "hitran_cia")
    opacity_folder = os.path.join(helios_path, "input", "opacity", "r50_kdistr") + "/"
    output_dir = os.path.join(os.path.dirname(__file__), "images", "CIAreport")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Find all sources for this pair
    print(f"\n{'='*70}")
    print(f"Searching for CIA opacity sources for pair: {args.pair}")
    print(f"{'='*70}")

    sources = find_cia_sources_for_pair(args.pair, hitran_dir)

    if not sources:
        print(f"ERROR: No CIA opacity sources found for pair {args.pair} in {hitran_dir}")
        sys.exit(1)

    print(f"\nFound {len(sources)} source(s):")
    for source_name, filepath in sources:
        print(f"  - {source_name}")

    # Identify default source
    default_source_name = DEFAULT_CIA_SOURCES.get(args.pair)
    if default_source_name is None:
        print(f"\nWarning: No default source defined for pair {args.pair}")
        print(f"Using first source as default: {sources[0][0]}")
        default_source_name = sources[0][0]
    else:
        print(f"\nDefault source for {args.pair}: {default_source_name}")

    # Find default source in list
    default_source_path = None
    for source_name, filepath in sources:
        if source_name == default_source_name:
            default_source_path = filepath
            break

    if default_source_path is None:
        print(f"Warning: Default source {default_source_name} not found in available sources")
        print(f"Using first source as default: {sources[0][0]}")
        default_source_name, default_source_path = sources[0]

    # ========================================================================
    # LOAD BASE OPACITIES ONCE (Efficient!)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Loading BASE opacities (individual absorbers + non-target CIAs)")
    print(f"{'='*70}")

    base_quant, mu_2d, mr_1_tiled, mr_2_tiled, mu_tiled = load_base_opacity(
        args.pair, opacity_folder
    )

    # Finalize base quant for use as "without CIA" comparison
    quant_without = finalize_base_quant(base_quant)

    # Build the default source with CIA (for difference plots)
    print(f"\n{'='*70}")
    print(f"Building DEFAULT source WITH target CIA: {default_source_name}")
    print(f"{'='*70}")
    quant_default_with = build_quant_with_cia(
        base_quant, default_source_path, args.pair,
        mr_1_tiled, mr_2_tiled, mu_tiled
    )

    T_grid = np.array(base_quant.ktemp)
    P_grid = np.array(base_quant.kpress)

    # ========================================================================
    # PROCESS EACH SOURCE (Reusing base opacities!)
    # ========================================================================
    for i, (source_name, source_path) in enumerate(sources):
        print(f"\n{'='*70}")
        print(f"Processing source {i+1}/{len(sources)}: {source_name}")
        print(f"{'='*70}")

        # Build this source with CIA on top of the SAME base
        print(f"\nBuilding source WITH target CIA...")
        quant_source_with = build_quant_with_cia(
            base_quant, source_path, args.pair,
            mr_1_tiled, mr_2_tiled, mu_tiled
        )

        # Generate relevance plot (source vs no-CIA baseline)
        print(f"\nGenerating relevance plot...")
        plot_relevance_map(quant_source_with, quant_without, args.pair, source_name,
                          args.Tmax, output_dir, args.alpha, mu_2d, T_grid, P_grid,
                          fine_Tstep=args.fine_Tstep)

        # Generate difference plot (only if not default source)
        if source_name != default_source_name:
            print(f"\nGenerating difference plot vs default...")
            plot_difference_map(quant_source_with, quant_default_with, args.pair,
                               source_name, default_source_name, args.Tmax, output_dir,
                               args.alpha, mu_2d, T_grid, P_grid,
                               fine_Tstep=args.fine_Tstep)

    print(f"\n{'='*70}")
    print(f"All done! Generated {len(sources)} relevance plot(s)")
    if len(sources) > 1:
        print(f"and {len(sources)-1} difference plot(s)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
