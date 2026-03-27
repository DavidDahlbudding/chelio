import logging
import os
import shutil
import sys
import time
from math import ceil
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

# This import is dependent on the HELIOS_PATH environment variable.
# The main run_coupled.py script will ensure this is set.
try:
    sys.path.append(os.path.join(os.environ["HELIOS_PATH"], "source"))
    from species_database import species_lib
except (ImportError, KeyError):
    log.error(
        "Could not import species_database. Please set the HELIOS_PATH environment variable."
    )
    raise

try:
    sys.path.append(os.path.join(os.environ["CHELIO_PATH"], "chelio_sim"))
    from rt_utils import parse_mixfile
except (ImportError, KeyError):
    log.error(
        "Could not import species_database. Please set the HELIOS_PATH environment variable."
    )
    raise

names = np.array(['H2', 'He', 'N2', 'CH4', 'O2', 'CO2', 'H2O', 'NH3', 'CO'])
masses = np.array([2.016, 4.0026, 28.0134, 16.0425, 31.9988, 44.0095, 18.0153, 17.031, 28.010]) # amu
# from https://en.wikipedia.org/wiki/Triple_point
triple_Ts = np.array([13.8033, 2.1768, 63.18, 90.68, 54.36, 216.55, 273.16, 195.4, 68.1]) # K
triple_Ps = np.array([7.04e3, 5.048e3, 12.6e3, 11.7e3, 0.144e3, 517e3, 0.611657e3, 6.06e3, 15.37e3]) # Pa
triple_Ps *= 1e1 # to dyn/cm^2
# from https://en.wikipedia.org/wiki/Critical_point_(thermodynamics)
critical_Ts = np.array([33.20, 5.19, 126.2, 190.8, 154.33, 304.19, 647.1, 405.5, 133.16]) # K
critical_Ps = np.array([1.300e6, 0.227e6, 3.39e6, 4.64e6, 5.043e6, 7.38e6, 22.06e6, 11.28e6, 3.498e6]) # Pa
critical_Ps *= 1e1 # to dyn/cm^2

mol_dict = {name: {'mass': mass, 'triple': [Tt, Pt], 'critical': [Tc, Pc]} for name, mass, Tt, Pt, Tc, Pc in zip(names, masses, triple_Ts, triple_Ps, critical_Ts, critical_Ps)}

# psat from GGchem
mmHg = 1.3328e+03 # dyn/cm^2
bar = 1.0e+06 # dyn/cm^2
kB = 1.380649e-16 # erg/K
R = 8.314462618e7 # erg/(K mol)

_J_TO_ERG = 1e7

# Default location for HELIOS pre-tabulated kappa/delad + c_p table.
# We overwrite this file for each iteration/run; HELIOS will write the 1D profile in the run output.
DEFAULT_DELAD_TABLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../helios_inputs/delad_chelio.dat")
)

_CP_INTERP_CACHE = {}

# Species for which p_sat() returns a valid physical formula.
# He returns NaN (fine); unknown species return np.empty_like (uninitialized garbage - latent bug).
# Using an allowlist avoids calling p_sat for non-condensing or unknown species.
_PSAT_FORMULA_SPECIES = frozenset(['H2', 'N2', 'CH4', 'O2', 'CO2', 'H2O', 'NH3', 'CO'])


def _validate_log10_pressure_grid(p_bar: np.ndarray, rtol: float = 1e-2, atol: float = 1e-1) -> None:
    """Validate that log10(P) spacing is constant (HELIOS requirement for delad tables)."""
    if p_bar.ndim != 1:
        raise ValueError("Pressure grid must be 1D")
    if len(p_bar) < 2:
        raise ValueError("Pressure grid must have at least 2 points")
    if np.any(~np.isfinite(p_bar)) or np.any(p_bar <= 0):
        raise ValueError("Pressure grid must be finite and strictly positive")
    logp = np.log10(p_bar)
    dlogp = np.diff(logp)
    if not np.allclose(dlogp, dlogp[0], rtol=rtol, atol=atol):
        print("max log10(P) differences:", np.abs(dlogp - dlogp[0]).max(), "at:", logp[np.abs(dlogp - dlogp[0]).argmax()])
        raise ValueError(
            "Pressure grid must have constant spacing in log10(P) for HELIOS kappa/delad tables"
        )


def _get_janaf_cp_interpolator(species: str):
    """Return Cp(T) interpolator from GGchem JANAF tables.

    Expected JANAF format: skip 3 header rows; first two columns are T[K] and Cp[J mol^-1 K^-1].
    """
    if species in _CP_INTERP_CACHE:
        return _CP_INTERP_CACHE[species]

    ggchem_path = os.environ.get("GGCHEM_PATH")
    if not ggchem_path:
        raise KeyError("GGCHEM_PATH environment variable not set")

    janaf_file = os.path.join(ggchem_path, "data", "JANAF", f"{species}.txt")
    cp_data = np.loadtxt(janaf_file, skiprows=3)[:, :2]
    interp = interp1d(cp_data[:, 0], cp_data[:, 1], bounds_error=False, fill_value=(cp_data[0, 1], cp_data[-1, 1]))
    _CP_INTERP_CACHE[species] = interp
    return interp


def write_helios_delad_table(
    p_bar: np.ndarray,
    t_profile_k: np.ndarray,
    species: Sequence[str],
    vmr_profile: np.ndarray,
    out_path: str = DEFAULT_DELAD_TABLE_PATH,
    t_step_max_k: float = 20.0,
    ignore_missing_cp_below_vmr: float = 1e-20,
) -> str:
    """Write a HELIOS standard-format pre-tabulated kappa/delad (+ c_p) file.

    Inputs:
    - p_bar: 1D pressure grid [bar], must be constant spacing in log10(P)
    - t_profile_k: 1D temperature profile on the same pressure grid [K]
    - species: list of species names matching columns in vmr_profile
    - vmr_profile: 2D array shape (nP, nSpecies), assumed to be mole fractions (VMR)

    Output file format must match HELIOS' example (2-line header, then columns):
    temp.[K]  press.[1e-6bar]  delad  c_p[erg mol^-1 K^-1]  log_S[log10(erg g^-1 K^-1)]

    Notes:
    - Pressure column is in 1e-6 bar units (= dyn/cm^2), i.e. P_out = P_bar * 1e6.
    - c_p is mixture molar heat capacity in cgs (erg mol^-1 K^-1), using JANAF Cp in J mol^-1 K^-1.
    - delad (kappa) is computed as R_universal / c_p (dimensionless).
    - entropy column is written as 0 (optional in HELIOS and unused by RT).
    """
    p_bar = np.asarray(p_bar, dtype=float)
    t_profile_k = np.asarray(t_profile_k, dtype=float)
    vmr_profile = np.asarray(vmr_profile, dtype=float)

    _validate_log10_pressure_grid(p_bar)

    if t_profile_k.ndim != 1 or len(t_profile_k) != len(p_bar):
        raise ValueError("t_profile_k must be 1D and same length as p_bar")
    if vmr_profile.ndim != 2 or vmr_profile.shape[0] != len(p_bar) or vmr_profile.shape[1] != len(species):
        raise ValueError("vmr_profile must have shape (nP, nSpecies)")

    # Build linear T grid with constant steps, step size <= t_step_max_k.
    t_min = float(np.nanmin(t_profile_k)) * 0.9
    t_max = float(np.nanmax(t_profile_k)) * 1.1
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Temperature profile contains non-finite values")
    if t_max < t_min:
        t_min, t_max = t_max, t_min
    n_intervals = int(ceil(max(t_max - t_min, 0.0) / float(t_step_max_k)))
    n_intervals = max(n_intervals, 1)
    t_grid = np.linspace(t_min, t_max, n_intervals + 1)

    # Normalize VMRs (mole fractions) per layer to avoid small GGchem truncation effects.
    vmr = np.clip(vmr_profile, 0.0, np.inf)
    vmr_sum = vmr.sum(axis=1)
    if np.any(vmr_sum <= 0):
        raise ValueError("VMR profile sums to <= 0 in at least one layer")
    vmr = vmr / vmr_sum[:, np.newaxis]

    # Load Cp interpolators; allow skipping species with tiny VMR if Cp file missing.
    kept_species = []
    kept_vmr_cols = []
    kept_cp = []
    for j, s in enumerate(species):
        if s.startswith("CIA"):
            # CIA entries are not real gas species for thermodynamics.
            continue

        max_vmr = float(np.nanmax(vmr[:, j]))
        try:
            cp_interp = _get_janaf_cp_interpolator(s)
        except Exception as e:
            if max_vmr < ignore_missing_cp_below_vmr:
                log.debug(f"Skipping Cp for trace species '{s}' (max VMR {max_vmr:.3e}): {e}")
                continue
            raise RuntimeError(
                f"Could not load JANAF Cp data for species '{s}' (needed for kappa table). File expected at "
                f"$GGCHEM_PATH/data/JANAF/{s}.txt"
            ) from e

        kept_species.append(s)
        kept_vmr_cols.append(j)
        kept_cp.append(cp_interp)

    if not kept_species:
        raise RuntimeError("No species with Cp data available to compute kappa/delad")

    vmr_kept = vmr[:, kept_vmr_cols]
    vmr_kept_sum = vmr_kept.sum(axis=1)
    if np.any(vmr_kept_sum <= 0):
        raise RuntimeError("After filtering, VMR sums to <= 0 in at least one layer")
    vmr_kept = vmr_kept / vmr_kept_sum[:, np.newaxis]

    # --- Compute vmr_grid: shape (nP, nT, nS) ---
    # Apply condensation limits at each (P, T) grid point using p_sat(),
    # starting from the 1-D profile VMRs as the base mixing ratios.
    nP_g = len(p_bar)
    nT_g = len(t_grid)
    nS_g = len(kept_species)
    p_bar_dyn = p_bar * 1e6  # bar -> dyn/cm^2 (same units as p_sat return value)

    # A. Saturation VMR ceiling: max_vmr_grid[iP, iT, s] = p_sat(T[iT], s) / P[iP]
    #    inf => no condensation limit (non-condensing or unknown species).
    max_vmr_grid = np.full((nP_g, nT_g, nS_g), np.inf)
    for s_idx, s in enumerate(kept_species):
        if s not in _PSAT_FORMULA_SPECIES:
            continue  # He, CIA-like, or unknown: no condensation limit
        psat_t = p_sat(t_grid, s)  # (nT,) dyn/cm^2; formula extrapolates all T
        psat_t = np.where(np.isnan(psat_t), np.inf, psat_t)
        # broadcast (nT,) / (nP,) -> (nP, nT)
        max_vmr_grid[:, :, s_idx] = psat_t[np.newaxis, :] / p_bar_dyn[:, np.newaxis]

    # B. Initialise from 1-D profile VMRs, broadcast over T dimension
    vmr_grid = np.broadcast_to(
        vmr_kept[:, np.newaxis, :], (nP_g, nT_g, nS_g)
    ).copy()

    # C. Iterative condensation redistribution (vectorised over nP x nT)
    #    Same algorithm as create_constant_mixfile, with species on axis=2.
    for _cond_iter in range(100):
        prev_vmr = vmr_grid.copy()
        limited = vmr_grid >= max_vmr_grid
        vmr_grid = np.where(limited, max_vmr_grid, vmr_grid)

        current_sum = vmr_grid.sum(axis=2, keepdims=True)  # (nP, nT, 1)
        deficit = 1.0 - current_sum
        if np.all(np.abs(deficit) < 1e-12):
            break

        can_accept = (~limited) & (vmr_grid >= 1e-29)
        if not np.any(can_accept):
            log.warning(
                "write_helios_delad_table: all kept species hit condensation ceiling "
                "at some (P, T) grid point(s). delad may be inaccurate there."
            )
            break

        weights = prev_vmr * can_accept
        weight_sum = weights.sum(axis=2, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        vmr_grid += (weights / weight_sum) * deficit
        vmr_grid = np.minimum(vmr_grid, max_vmr_grid)  # re-clamp after redistribution

        if np.allclose(vmr_grid - prev_vmr, 0.0, atol=1e-12):
            break

    # D. Normalise per (P, T) point so delad remains physical when condensation
    #    depletes part of the gas (avoid unphysically small delad from partial sums).
    vmr_grid_sum = vmr_grid.sum(axis=2, keepdims=True)
    vmr_grid_sum = np.where(vmr_grid_sum <= 0.0, 1.0, vmr_grid_sum)
    vmr_grid = vmr_grid / vmr_grid_sum

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    header_1 = (
        "This file contains the adiabatic coefficient (kappa, delad), heat capacity and entropy per unit mass"
    )
    header_2 = (
        "temp.[K]  press.[1e-6bar]  delad           c_p[erg mol^-1 K^-1]    log_S[log10(erg g^-1 K^-1)]"
    )

    p_out = p_bar * 1e6  # bar -> 1e-6 bar units (= dyn/cm^2)

    with open(out_path, "w") as f:
        f.write(header_1 + "\n")
        f.write(header_2 + "\n")

        # Stream by temperature slice to keep memory bounded.
        for iT, T in enumerate(t_grid):
            cp_species_j = np.array([cp(T) for cp in kept_cp], dtype=float)  # J mol^-1 K^-1
            cp_mix_j = vmr_grid[:, iT, :] @ cp_species_j  # (nP,)
            cp_mix_erg = cp_mix_j * _J_TO_ERG
            if np.any(cp_mix_erg <= 0) or np.any(~np.isfinite(cp_mix_erg)):
                raise RuntimeError("Non-physical mixture Cp encountered while building kappa table")

            delad = R / cp_mix_erg

            for P_val, delad_val, cp_val in zip(p_out, delad, cp_mix_erg):
                f.write(
                    f"{T:.6f}  {P_val:.6e}  {delad_val:.6e}  {cp_val:.6e}  0\n"
                )

    log.info(f"Wrote HELIOS kappa/delad table to '{out_path}'")
    return out_path

def p_sat(T, species, mask=False):
    """
    Calculates the saturation pressure of a given species at a given temperature.

    Args:
        T (np.ndarray): Temperature (K)
        species (str): Species name
        mask (bool): If True, the saturation pressure is set to NaN for temperatures outside the range of the species.

    Returns:
        np.ndarray: Saturation pressure (dyn/cm^2 = 1e6 bar)
    """

    if species == 'H2':
        # NIST (21-32K)
        psat = 10.0**(3.54314 - 99.395/(T + 7.726))*bar
        if mask:
            mask = (T < 21.0) | (T > 32.0)
            psat[mask] = np.nan
        return psat

    elif species == 'He':
        psat = np.empty_like(T)
        psat.fill(np.nan)  # He does not have a saturation pressure in the range
        return psat
         
    elif species == 'N2':
        # NIST (63-126K)
        psat = 10.0**(3.7362 - 264.651/(T - 6.788))*bar
        if mask:
            mask = (T < 63.0) | (T > 126.0)
            psat[mask] = np.nan
        return psat

    elif species == 'O2':
        # NIST (54-154K)
        psat = 10.0**(3.9523 - 340.024/(T - 4.144))*bar
        # psat = 10.0**(3.85845 - 325.675/(T - 5.667))*bar # (54-100K)
        if mask:
            mask = (T < 54.0) | (T > 154.0)
            psat[mask] = np.nan
        return psat

    elif species == 'CH4':
        # only solid
        # NIST (90-190K) and GGchem
        # Prydz, R.; Goodwin, R.D., J. Chem. Thermodyn., 1972, 4,1 
        psat = 10.0**(3.9895 - 443.028/(T-0.49))*bar
        if mask:
            mask = (T < 90.0) | (T > 190.0)
            psat[mask] = np.nan
        return psat
    
    elif species == 'CO2':
        # Yaws' Chemical Properties Handbook (McGraw-Hill 1999) (216-305K), GGchem
        C = np.array([35.0187E+00, -1.5119E+03, -1.1335E+01, 9.3383E-03, 7.7626E-10])
        psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
        psat = 10**psat * mmHg
        if mask:
            mask = (T < 216.0) | (T > 305.0)
            psat[mask] = np.nan
        return psat

    elif species == 'H2O':
        
            # Ackerman & Marley 2001 (liquid!), GGchem
            # TC = T - 273.15
            # psat_l = 6112.1*np.exp((18.729*TC - TC**2/227.3)/(TC + 257.87))
            # Yaws' Chemical Properties Handbook (McGraw-Hill 1999) (273-647K)
            C = np.array([29.8605e+00, -3.1522e+03, -7.3037e+00, 2.4247e-09, 1.8090e-06])
            psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
            psat = 10**psat * mmHg
            psat_final = psat.copy()
    
            # Ackerman & Marley 2001 (solid), GGchem
            TC   = np.minimum(2000.0,T)-273.15 # T[degree Celsius]
            psat = 6111.5*np.exp((23.036*TC - TC**2/333.7)/(TC + 279.82))
            psat_final[T<273.16] = psat[T<273.16]

            # Haldemann+ 2020
            #psat = 0.05 + 2.95 * (T/1000 - 1)/39 # GPa
            #psat = psat * 1e+10 # dyn/cm^2
            #psat_final[T>647.0] = psat[T>647.0]

            if mask:
                mask = T > 647.0
                psat_final[mask] = np.nan

            return psat_final
    
    elif species == 'NH3':
        # CRC Handbook of Chemistry and Physics (Weast 1971), GGchem
        psat = np.exp(10.53 - 2161.0/T - 86596.0/T**2)*bar
        if mask:
            mask = T > mols['NH3']['critical'][0]  # 405.5 K
            psat[mask] = np.nan

        # NIST (164-371.5K)
        # psat = 10.0**(3.18757 - 506.713/(T - 80.78))*bar
        # psat[T>239.6] = 10**(4.86886 - 1113.928/(T[T>239.6] - 10.409))*bar
        # if mask:
        #     mask = (T < 164.0) | (T > 371.5)
        #     psat[mask] = np.nan
        
        return psat
    
    elif species == 'CO':
        # Yaws' Chemical Properties Handbook (McGraw-Hill 1999) (68-132K), GGchem
        C = np.array([51.8145E+00, -7.8824E+02, -2.2734E+01, 5.1225E-02, 4.6603E-11])
        psat = C[0] + C[1] / T + C[2] * np.log10(T) + C[3] * T + C[4] * T**2
        psat = 10**psat * mmHg
        if mask:
            mask = (T < 68.0) | (T > 132.0)
            psat[mask] = np.nan
        return psat

    else:
        # warning
        log.warning(f"Saturation pressure for species {species} not found. Returning NaN.")
        return np.empty_like(T)


def append_profiles(header, data, ref_pt=os.path.join(os.environ["GGCHEM_PATH"], "structures", "pt_helios.in")):
    """
    Appends the data with the last step of itself (+ condensation) on the grid of the reference T-P profile.

    Args:
        header (np.ndarray): Header of the Vertical Mixfile output file
        data (np.ndarray): Data of the Vertical Mixfile output file
        ref_pt (str): Path to the reference P-T profile

    Returns:
        np.ndarray: Appended data
    """
    # read reference T-P profile
    ref_pt = np.loadtxt(ref_pt, skiprows=1, dtype=float)
    ref_P = ref_pt[:,0] # bar
    ref_T = ref_pt[:,1] # K

    missing_data = np.zeros((len(ref_P)-len(data), len(data[0])))
    max_vmr = np.zeros((len(ref_P)-len(data), len(data[0])-5))

    i_append = len(data)

    for i in range(max_vmr.shape[1]):
        max_vmr[:,i] = p_sat(ref_T[i_append:], header[i+5]) * 1e-6 # dyn/cm^2 -> bar
    max_vmr = max_vmr / ref_P[i_append:,np.newaxis]

    i_h2 = np.where(header == "H2")[0][0] - 5
    max_vmr[:,i_h2] = 1.01 # H2 condensation is ignored

    missing_data[:,0] = ref_P[i_append:]
    missing_data[:,1] = ref_T[i_append:]

    missing_data[:,2] = (missing_data[:,0] * 1e6) / (kB * missing_data[:,1]) # n_tot (cm^-3)
    # mean molecular weight mu gets calculated later
    missing_data[:,4] = data[-1,4] # electron VMR (usually < 1e-300)

    missing_data[:,5:] = np.minimum(max_vmr, data[-1,5:][np.newaxis,:])

    total_vmr = np.sum(missing_data[:,5:], axis=-1)
    while np.any(total_vmr < (1 - 1e-3)):
        mask = missing_data[:,5:] < max_vmr
        
        distribution_factor = missing_data[:,5:].copy() # distribute proportional to the "current" VMR
        distribution_factor[~mask] = 0.0 # don't distribut missing VMR to already saturated species
        distribution_factor = distribution_factor / np.sum(distribution_factor, axis=-1, keepdims=True) # normalize

        missing_data[:,5:] = missing_data[:,5:] + (distribution_factor * (1 - total_vmr)[:,np.newaxis])
        missing_data[:,5:] = np.minimum(max_vmr, missing_data[:,5:])
        total_vmr = np.sum(missing_data[:,5:], axis=-1)

    if np.any(missing_data[:,i_h2+5] > max_vmr[:,i_h2]):
        # throw warning
        log.warning("H2 above saturation pressure!")
        time.sleep(0.5)

    # calculate mean molecular weight
    mu = np.array([species_lib[s].weight for s in header[5:] if s in species_lib.keys()])
    mu = np.sum(missing_data[:,5:] * mu, axis=-1)
    missing_data[:,3] = mu

    missing_data = np.concatenate((data, missing_data), axis=0)

    return missing_data


def create_constant_mixfile(p_bar, T_k, mixing_ratios, helios_mixfile_path, relative_humidity=1.0, coupling_speed_up=False):
    """
    Creates a HELIOS mixfile with constant mixing ratios, limited by condensation.

    Args:
        p_bar (np.ndarray): Pressure grid in bar
        T_k (np.ndarray): Temperature grid in K
        mixing_ratios (dict): Species name -> constant mixing ratio
                              (e.g., {"H2O": 0.01, "CO2": 0.001, "N2": 0.989})
        helios_mixfile_path (str): Output path
        relative_humidity (float): Fractional relative humidity applied to H2O saturation cap
                                   (e.g., 0.8 means H2O VMR <= 0.8 * p_sat(T)/P). Default: 1.0.
    """
    log.info(f"Creating constant-VMR mixfile at '{helios_mixfile_path}'")
    
    species_list = list(mixing_ratios.keys())
    n_layers = len(p_bar)
    n_species = len(species_list)
    
    # 1. Pre-calculate Saturation Limits (Vectorized)
    # Shape: (n_layers, n_species)
    max_vmr = np.full((n_layers, n_species), np.inf)
    for i, s in enumerate(species_list):
        psat = p_sat(T_k, s) # dyn/cm2
        if not np.all(np.isnan(psat)):
            # Convert dyn/cm2 to bar: 1 bar = 1e6 dyn/cm2
            rh_factor = relative_humidity if s == "H2O" else 1.0
            max_vmr[:, i] = rh_factor * (psat * 1e-6) / p_bar

    # 2. Initial VMRs (normalized)
    vmr = np.array([mixing_ratios[s] for s in species_list])
    vmr = np.tile(vmr, (n_layers, 1))
    vmr /= vmr.sum(axis=1, keepdims=True)

    # 3. Iterative Adjustment
    max_iter = 100
    for _ in range(max_iter):
        prev_vmr = vmr.copy()
        
        # Apply limits
        #print(vmr[0,:])
        #print(max_vmr[0,:])
        limited = vmr >= max_vmr
        vmr[limited] = max_vmr[limited]
        
        # Calculate deficit
        current_sum = vmr.sum(axis=1, keepdims=True)
        deficit = 1.0 - current_sum
        
        # If no deficit (or converged), break
        if np.all(np.isclose(current_sum, 1.0, atol=1e-12)):
            break
            
        # Identify species that can still accept gas
        can_accept = np.logical_and(~limited, vmr >= 1e-29)
        if not np.any(can_accept):
            # Physical failure: everything is condensed
            log.warning("All species saturated! Mean molecular weight may be inaccurate.")
            break
            
        # Redistribute deficit proportionally to those NOT limited
        # Mask weights for limited species
        weights = prev_vmr * can_accept
        weight_sum = weights.sum(axis=1, keepdims=True)
        
        # Avoid division by zero if all non-limited have 0 initial VMR
        weight_sum[weight_sum == 0] = 1.0 
        
        vmr += (weights / weight_sum) * deficit

        if np.allclose((vmr-prev_vmr), 0.0, atol=1e-12):
            break

    # 4. Mean Molecular Weight Calculation
    # Using your renormalization approach:
    actual_sum = vmr.sum(axis=1)
    log.info(actual_sum.min())
    mu_weights = np.array([species_lib[s].weight if s in species_lib else 2.0 for s in species_list])
    mean_mu = np.sum(vmr * mu_weights, axis=1)


    # WHAT IF VMR DOES NOT SUM TO 1 (all considered species condese)?
    # EITHER assume missing "invisible" species has e.g. MMW = 2.0:
    #mean_mu += (1-np.sum(vmr, axis=1)) * 2.0
    # OR "renormalize" for MMW calculation:
    mean_mu /= actual_sum #.flatten()?
    
    # Calculate total number density: n = P / (kB * T)
    n_tot = (p_bar * 1e6) / (kB * T_k)  # p in dyn/cm²
    
    # Build output array: P, T, n_tot, mu, e-, species...
    new_header = np.array(["P(bar)", "T(k)", "n_<tot>(cm-3)", "m(u)", "e-"] + species_list)
    new_data = np.zeros((n_layers, len(new_header)))
    new_data[:, 0] = p_bar
    new_data[:, 1] = T_k
    new_data[:, 2] = n_tot
    new_data[:, 3] = mean_mu
    new_data[:, 4] = 0.0  # electrons (negligible)
    new_data[:, 5:] = vmr

    if coupling_speed_up:
        # replace number i in helios_mixfile_path (between last "_" and ".dat") with "i-1"
        base, ext = os.path.splitext(helios_mixfile_path)
        if "_" in base:
            prefix, num_str = base.rsplit("_", 1)
            if num_str.isdigit():
                prev_num_str = str(int(num_str) - 1)
                prev_helios_mixfile_path = f"{prefix}_{prev_num_str}{ext}"
                if os.path.exists(prev_helios_mixfile_path):
                    p_prev, mu_prev, species_prev, vmr_prev = parse_mixfile(prev_helios_mixfile_path)
                    # Check if species_prev matches species_list
                    if set(species_prev) == set(species_list):
                        # average mu and (log) vmr with previous mixfile
                        new_data[:, 3] = (new_data[:, 3] + mu_prev) / 2.0
                        for s in species_list:
                            idx_new = species_list.index(s)
                            zero_mask = vmr_prev[s] != 0
                            new_data[~zero_mask, 5 + idx_new] = 10**(np.log10(new_data[~zero_mask, 5 + idx_new]) + np.log10(vmr_prev[s][~zero_mask]) / 2.0)
                            new_data[zero_mask, 5 + idx_new] = new_data[zero_mask, 5 + idx_new]
                else:
                    log.warning(f"Previous mixfile '{prev_helios_mixfile_path}' not found for speed-up. Generating new mixfile.")
            else:
                log.warning(f"Filename '{helios_mixfile_path}' does not end with a number for speed-up. Generating new mixfile.")
        else:
            log.warning(f"Filename '{helios_mixfile_path}' does not contain '_' for speed-up. Generating new mixfile.")

    # Generate pre-tabulated kappa/delad + c_p table for HELIOS convection.
    # Written to a shared location and overwritten each iteration.
    write_helios_delad_table(
        p_bar=p_bar,
        t_profile_k=T_k,
        species=species_list,
        vmr_profile=vmr,
        out_path=DEFAULT_DELAD_TABLE_PATH,
    )
    
    # Format header nicely (same as convert_ggchem_to_helios)
    header_string = []
    for i in range(len(new_header)):
        header_string.append(new_header[i])
        n_spaces = 16 - len(new_header[i])
        header_string.append(n_spaces * " " + "\t")
    header_string = "".join(header_string[:-1])
    
    # Save to file
    try:
        np.savetxt(
            helios_mixfile_path,
            new_data,
            header=header_string,
            fmt="%.10e",
            comments="",
            delimiter="\t",
        )
        log.info(f"Successfully created constant-VMR mixfile at {helios_mixfile_path}")
    except IOError as e:
        log.error(f"Failed to write mixfile: {e}")
        raise


def convert_ggchem_to_helios(ggchem_output_path, helios_mixfile_path, ref_pt=os.path.join(os.environ["GGCHEM_PATH"], "structures", "pt_helios.in"), coupling_speed_up=False):
    """
    Converts GGchem output (Static_Conc.dat) to a HELIOS mixfile.

    Args:
        ggchem_output_path (str): Path to the GGchem output file (e.g., Static_Conc.dat).
        helios_mixfile_path (str): Path to write the output HELIOS mixfile to.
        coupling_speed_up (bool): Whether to use speed-up coupling with previous mixfile.
    """
    log.info(
        f"Converting GGchem output '{ggchem_output_path}' to HELIOS mixfile '{helios_mixfile_path}'"
    )

    # read relevant species from helios_inputs/species.dat
    # Path is relative to this file's location in the package structure.
    species_dat_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../helios_inputs/species.dat")
    )
    try:
        species = np.loadtxt(species_dat_path, dtype=str, usecols=(0,))[1:]
    except FileNotFoundError:
        log.error(f"species.dat file not found at {species_dat_path}")
        raise

    # if CIA_N2N2 exists, append N2 to species
    if "CIA_N2N2" in species:
        species = np.append(species, "N2")

    # remove CIA (assumes that CIA species, other than N2, are already accounted for!)
    species = np.array([s for s in species if s[:3] != "CIA"])

    # read GGchem output file
    try:
        header = np.loadtxt(ggchem_output_path, skiprows=2, max_rows=1, dtype=str)
        dimension = np.genfromtxt(
            ggchem_output_path, dtype=int, max_rows=1, skip_header=1
        )
        data = np.loadtxt(ggchem_output_path, skiprows=3)
    except FileNotFoundError:
        log.error(f"GGchem output file not found at {ggchem_output_path}")
        raise
    except Exception as e:
        log.error(f"Error reading GGchem output file: {e}")
        raise

    n_elem = dimension[0]
    n_mol = dimension[1]
    # n_dust = dimension[2]
    n_layers = dimension[3]

    conversions = {
        "P(bar)": "pgas",
        "T(k)": "Tg",
        "n_<tot>(cm-3)": "calculated_ntot",
        "m(u)": "calculated_mu",
        "e-": "el",
    }

    # create header with correct (species) names
    new_header = list(conversions.keys())
    new_header.extend(
        [
            species_lib[s].name
            for s in header[4 : 4 + n_elem + n_mol]
            if s in species
        ]
    )
    new_header = np.array(new_header)

    # convert data
    new_data = np.zeros((len(data), len(new_header)))

    # Pressure (convert from cgs (dyn/cm^2) to bar)
    new_data[:, 0] = (
        data[:, np.where(header == conversions[new_header[0]])[0][0]] * 1e-6
    )

    # Temperature
    new_data[:, 1] = data[:, np.where(header == conversions[new_header[1]])[0][0]]

    # Calculate total number density
    n_tot = 10 ** data[:, 3 : 4 + n_elem + n_mol]
    n_tot = np.sum(n_tot, axis=1)
    new_data[:, 2] = n_tot

    # Calculate mean molecular weight
    mu = np.zeros(len(data))
    a_tot = np.zeros(len(data))

    for i, s in enumerate(header[3 : 4 + n_elem + n_mol]):
        if s in species_lib.keys():
            a_mol = 10 ** data[:, 3 + i]  # number density
            a_mol = a_mol / n_tot  # fraction
            mu += a_mol * species_lib[s].weight
            a_tot += a_mol

        elif s == "el":
            s = "e-"
            a_mol = 10 ** data[:, 3]  # number density
            a_mol = a_mol / n_tot  # fraction
            mu += a_mol * species_lib[s].weight
            a_tot += a_mol

        # save species fractions
        if s in species or s == "e-":
            new_data[:, np.where(new_header == species_lib[s].name)[0][0]] = a_mol

    if np.any(a_tot < 0.99):
        log.warning("sum of considered species fractions is less than 1 in some layers!")
        time.sleep(0.5)

    new_data[:, 3] = mu

    if n_layers > len(data):
        new_data = append_profiles(new_header, new_data, ref_pt=ref_pt)

    if coupling_speed_up:
        # replace number i in helios_mixfile_path (between last "_" and ".dat") with "i-1"
        base, ext = os.path.splitext(helios_mixfile_path)
        if "_" in base:
            prefix, num_str = base.rsplit("_", 1)
            if num_str.isdigit():
                prev_num_str = str(int(num_str) - 1)
                prev_helios_mixfile_path = f"{prefix}_{prev_num_str}{ext}"
                if os.path.exists(prev_helios_mixfile_path):
                    p_prev, mu_prev, species_prev, vmr_prev = parse_mixfile(prev_helios_mixfile_path)
                    # Check if species_prev matches species_list
                    if set(species_prev) == set(species):
                        # average mu and (log) vmr with previous mixfile
                        new_data[:, 3] = (new_data[:, 3] + mu_prev) / 2.0
                        for s in species:
                            idx_new = species.index(s)
                            zero_mask = vmr_prev[s] != 0
                            new_data[~zero_mask, 5 + idx_new] = 10**(np.log10(new_data[~zero_mask, 5 + idx_new]) + np.log10(vmr_prev[s][~zero_mask]) / 2.0)
                            new_data[zero_mask, 5 + idx_new] = new_data[zero_mask, 5 + idx_new]
                else:
                    log.warning(f"Previous mixfile '{prev_helios_mixfile_path}' not found for speed-up. Generating new mixfile.")
            else:
                log.warning(f"Filename '{helios_mixfile_path}' does not end with a number for speed-up. Generating new mixfile.")
        else:
            log.warning(f"Filename '{helios_mixfile_path}' does not contain '_' for speed-up. Generating new mixfile.")

    # Generate pre-tabulated kappa/delad + c_p table for HELIOS convection.
    # Written to a shared location and overwritten each iteration.
    write_helios_delad_table(
        p_bar=new_data[:, 0],
        t_profile_k=new_data[:, 1],
        species=[str(s) for s in new_header[5:]],
        vmr_profile=new_data[:, 5:],
        out_path=DEFAULT_DELAD_TABLE_PATH,
    )

    # nicely format header
    header_string = []
    for i in range(len(new_header)):
        header_string.append(new_header[i])
        n_spaces = 16 - len(new_header[i])
        header_string.append(n_spaces * " " + "\t")
    header_string = "".join(header_string[:-1])

    # save to file
    try:
        np.savetxt(
            helios_mixfile_path,
            new_data,
            header=header_string,
            fmt="%.10e",
            comments="",
            delimiter="\t",
        )
        log.info(f"Successfully created HELIOS mixfile at {helios_mixfile_path}")
    except IOError as e:
        log.error(f"Failed to write mixfile: {e}")
        raise

if __name__ == "__main__":
    # This block allows for standalone testing of the conversion script.
    # It mimics the behavior of the original convert_mixfile.py script.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if "GGCHEM_PATH" not in os.environ or "HELIOS_PATH" not in os.environ:
        log.error("Please set GGCHEM_PATH and HELIOS_PATH environment variables for testing.")
        sys.exit(1)

    ggchem_file = os.path.join(os.environ["GGCHEM_PATH"], "Static_Conc.dat")

    if len(sys.argv) > 1:
        helios_file = sys.argv[1]
    else:
        # Default output for testing purposes
        helios_file = "test_mixfile_utils.dat"
        log.info(f"Output file not provided, writing to '{helios_file}'")

    convert_ggchem_to_helios(ggchem_file, helios_file)
