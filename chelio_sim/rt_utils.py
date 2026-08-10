import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# import pandas as pd # No longer needed

import sys
import os
helios_source = os.environ["HELIOS_PATH"] + "/source"
if helios_source not in sys.path:
    sys.path.append(helios_source)
from species_database import species_lib

# Physical constants in cgs units
H = 6.62607015e-27  # Planck constant (erg*s)
C = 2.99792458e10   # Speed of light (cm/s)
K_B = 1.380649e-16    # Boltzmann constant (erg/K)
AMU = 1.660539e-24    # Atomic mass unit (g)


def read_species_file(filepath):
    """
    Reads the species file and returns a list of species.
    """
    species = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('species') or line.startswith('\n'):
                continue
            parts = line.split()
            if parts[1] == 'yes':
                species.append(parts[0])
    species = np.array(species)

    return species

def read_opac_file(filepath):
    """
    Reads opacity data and grid parameters from an HDF5 file.
    Adapted from HELIOS source.
    """
    # reading opacity file
    print(f"Reading opacity file: {filepath}")
    with h5py.File(filepath, "r") as f:
        opac_k = f["kpoints"][:]
        wave = f["center wavelengths"][:]
        delta_wave = f["wavelength width of bins"][:]
        inter_wave = f["interface wavelengths"][:]
        gauss_y = f["ypoints"][:]
        ktemp = f["temperatures"][:]
        kpress = f["pressures"][:]
    return opac_k, wave, delta_wave, inter_wave, gauss_y, ktemp, kpress


def _planck(lam, temp):
    """
    Calculates the Planck function B(lambda, T).
    """
    term1 = 2 * H * C**2 / lam**5
    with np.errstate(over='ignore'):
        term2 = np.exp(H * C / (lam * K_B * temp)) - 1
    return term1 / term2


def _dplanck_dT(lam, T):
    """
    Calculates the derivative of the Planck function with respect to temperature.
    """
    term1 = 2 * H**2 * C**3 / (lam**6 * K_B * T**2)
    with np.errstate(over='ignore'):
        exp_term = np.exp(H * C / (lam * K_B * T))
        term2 = exp_term / (exp_term - 1)**2
    return term1 * term2


class OpacityCalculator:
    """
    Manages loading, interpolating, and calculating Rosseland mean opacities.
    """
    def __init__(self, species_list, opacity_paths, T_grid, P_grid, mode='lin'):
        self.species = species_list
        self.opacity_paths = opacity_paths
        self.T_grid = T_grid
        self.P_grid = P_grid
        self.opac_data = {}
        self.mode = mode
        self._load_and_interpolate_opacities()

    def _load_and_interpolate_opacities(self):
        """
        Loads and pre-interpolates opacity data for all species.
        """
        # Load data for the first species to get grid information
        first_species = self.species[0]
        (
            _, self.wave, self.delta_wave, self.inter_wave, self.gauss_y,
            _, _
        ) = read_opac_file(self.opacity_paths[first_species])
        
        self.gauss_weight = np.polynomial.legendre.leggauss(len(self.gauss_y))[1]

        for species in self.species:
            opac_k, _, _, _, _, ktemp, kpress = read_opac_file(self.opacity_paths[species])
            opac_k = opac_k.reshape(len(ktemp), len(kpress), len(self.wave), len(self.gauss_y))

            if self.mode == 'lin':
                opac_k = opac_k
            elif self.mode == 'log':
                opac_k = np.log10(opac_k)
            else:
                raise ValueError(f"Unknown mode '{self.mode}'; choose 'lin' or 'log'")
            
            # Create interpolator
            points = (ktemp, np.log10(kpress))
            self.opac_data[species] = {
                'interpolator': RegularGridInterpolator(points, opac_k, bounds_error=False, fill_value=None),
                'ktemp': ktemp,
                'kpress': kpress
            }
            
    def get_rosseland_mean(self, T, P, mixing_ratios, mus):
        """
        Calculates the Rosseland mean opacity for a given T, P(dyn/cm^2), composition (mixing ratios) and mean molecular weight (g/mol).
        """
        total_opac_k = np.zeros((len(self.wave), len(self.gauss_y)))
        
        for species in self.opac_data.keys():
            try:
                if species[:3] == 'CIA':
                    cia_pair = species_lib[species].fc_name.replace("1", "").split('&')
                    mix_ratio = mixing_ratios[cia_pair[0]] * mixing_ratios[cia_pair[1]]
                else:
                    mix_ratio = mixing_ratios[species]
            except KeyError:
                #print(f"Species {species} not found in mixing ratios")
                continue
            if mix_ratio > 0:
                mu_weight = species_lib[species].weight/mus
                interpolator = self.opac_data[species]['interpolator']
                # The interpolator expects a (2,) point for (T, P)
                T_bounded = np.maximum(T, 50) # avoid extrapolation to lower temperatures
                opac_values = interpolator([T_bounded, np.log10(P)])[0]
                if self.mode == 'lin':
                    opac_values = opac_values
                elif self.mode == 'log':
                    opac_values = 10**opac_values
                else:
                    raise ValueError(f"Unknown mode '{self.mode}'; choose 'lin' or 'log'")
                total_opac_k += opac_values * mu_weight * mix_ratio

        # Avoid division by zero
        total_opac_k[total_opac_k < 1e-100] = 1e-100

        return self._calculate_rosseland_mean(T, total_opac_k)

    def _calculate_rosseland_mean(self, T, opac_k):
        """
        Core calculation of the Rosseland mean opacity.
        """
        expanded_wave = self._expand_wave(self.inter_wave, self.gauss_y)
        dBdT = _dplanck_dT(expanded_wave, T)
        dBdT[np.isnan(dBdT)] = np.finfo(float).eps

        # Numerator of Rosseland mean integral
        num_integrated = np.sum(0.5 * self.delta_wave[:, None] * self.gauss_weight[None, :] * dBdT) #, axis=-1)
        numerator = np.sum(num_integrated)

        # Denominator of Rosseland mean integral
        denom_integrated = np.sum(0.5 * self.delta_wave[:, None] * self.gauss_weight[None, :] * dBdT / opac_k)#, axis=(-1, -2))
        denominator = np.sum(denom_integrated)
        
        if denominator == 0:
            return np.finfo(float).eps # return a small number if denominator is zero

        return numerator / denominator

    def _expand_wave(self, interwave, gauss_y):
        """
        Expand wavelength grid for Gaussian quadrature.
        """
        x = (gauss_y[None, :] - 0.5) * 2
        dwave = (interwave[1:] - interwave[:-1])[:, None]
        mean_wave = (interwave[1:] + interwave[:-1])[:, None] / 2
        return 0.5 * dwave * x + mean_wave


def parse_mixfile(mixfile_path):
    """
    Parses a HELIOS vertical mixfile using numpy.

    Args:
        mixfile_path (str): Path to the vertical_mix_*.dat file.

    Returns:
        tuple: Contains:
            - p_grid (np.ndarray): Pressure grid (bar).
            - mu_profile (np.ndarray): Mean molecular weight profile (g/mol).
            - species (list): List of species names.
            - mix_ratios (dict): Dictionary of mixing ratios for each species.
    """
    with open(mixfile_path, 'r') as f:
        header_line = f.readline().strip().replace('\t', '')
        # The header can contain multiple spaces or tabs as delimiters
        header = [item for item in header_line.split(' ') if item]

    data = np.loadtxt(mixfile_path, skiprows=1)

    # Find column indices from the header
    p_col_idx = header.index('P(bar)')
    mu_col_idx = header.index('m(u)')

    p_grid = data[:, p_col_idx]
    mu_profile = data[:, mu_col_idx]

    # Species are from the header, excluding known non-species columns
    non_species_cols = ['P(bar)', 'T(k)', 'n_<tot>(cm-3)', 'm(u)', 'e-']
    species = [s for s in header if s not in non_species_cols]

    mix_ratios = {}
    for s in species:
        s_col_idx = header.index(s)
        mix_ratios[s] = data[:, s_col_idx]

    return p_grid, mu_profile, species, mix_ratios

def scale_height(T, mu_val, g):
    """Scale height in cm"""
    return K_B * T / (mu_val * AMU * g)

def calculate_tp_profile(
    T_eff, p_surf, p_top, g, mu, nabla_ad,
    opacity_calculator, mix_ratios_profile, target_p_grid,
    D=1.66
):
    """
    Calculates the 1D temperature-pressure profile of an atmosphere.
    
    Uses altitude (z) as primary coordinate following tsurf.py approach.
    Integrates downward from top of atmosphere, calculating pressure from
    hydrostatic equilibrium and tracking optical depth.

    Args:
        T_eff (float): Effective temperature of the planet (K).
        p_surf (float): Surface pressure (bar).
        p_top (float): Pressure at the top of the atmosphere (bar).
        g (float): Surface gravity (cm/s^2).
        mu (callable): Function mu(P) that returns mean molecular weight (g/mol).
        nabla_ad (callable): Function nabla_ad(T, P) for the adiabatic gradient.
        opacity_calculator (OpacityCalculator): Initialized OpacityCalculator instance.
        mix_ratios_profile (callable): Function mix_ratios(P) that returns a dict
                                      of mixing ratios.
        target_p_grid (np.ndarray): The pressure grid for the final output (bar).
        D (float): The diffusion coefficient (default: 1.66).

    Returns:
        tuple: (interpolated_T, extra_info)
            - interpolated_T: Temperature profile on target_p_grid (K)
            - extra_info: dict with 'rosseland_mean', 'tau', 'z', 'T_r', 'p_r'
    """ 
    # Initial conditions at the top of the atmosphere
    T_rad = T_eff * (2**(-0.25))  # Radiative temperature at top
    
    # Get initial atmospheric properties
    mu_top = mu(p_top)
    
    # Initialize at the top of atmosphere
    p_rad = p_top
    T_prev = T_rad
    p_prev = p_rad
    optical_depth = 0.0
    z = 0.0
    
    # Flag to track if we've entered convective zone
    convective_switch = 0
    T_r = None  # Boundary temperature between radiative and convective
    p_r = None  # Boundary pressure between radiative and convective
    
    # Storage for profile
    z_profile = [z]
    p_profile = [p_rad]
    T_profile = [T_rad]
    tau_profile = [optical_depth]
    kappa_profile = [0.0]
    Delta_profile = [nabla_ad(T_rad, p_rad)]

    # Adaptive step size initialization
    delta_p_factor = 0.1
    i_iter = 0
    
    # Downward integration loop
    while p_prev < p_surf:
        # Get current atmospheric properties
        current_mu = mu(p_prev)
        current_mix_ratios = mix_ratios_profile(p_prev)
        
        # Radiative pressure at this altitude
        #p_rad = p_surf * np.exp(-z / scale_height(T_rad, current_mu, g))
        p_rad = p_prev * (1 + delta_p_factor)
        
        # Radiative temperature from optical depth
        T_rad = T_eff * ((0.5 * (1 + D * optical_depth))**0.25)
        
        delta_z = scale_height((T_rad+T_prev)/2, current_mu, g) * np.log(p_rad / p_prev)
        z = z - delta_z

        # Calculate radiative gradient d(ln T)/d(ln P)
        if p_rad > p_prev:
            grad = (np.log(T_rad) - np.log(T_prev)) / (np.log(p_rad) - np.log(p_prev))
        else:
            grad = 0.0
        
        # Get adiabatic gradient for comparison
        Delta = nabla_ad(T_prev, p_prev)
        
        # Check if we're in radiative or convective regime
        if grad < Delta:
            # Radiative zone
            kr = opacity_calculator.get_rosseland_mean(T_rad, p_rad * 1e6, current_mix_ratios, current_mu)
            #print('radiative', kr, T_rad, p_rad, current_mu)
            
            # Update optical depth
            d_tau = kr * (p_rad - p_prev) * 1e6 / g
            optical_depth = optical_depth + d_tau
            
            # Calculate optical depth gradient for adaptive stepping
            grad_opt = d_tau / delta_z
            
            T_prev = T_rad
            p_prev = p_rad
            
        else:
            # Convective zone
            if convective_switch == 0:
                # First time entering convective zone - mark boundary
                T_r = T_rad
                p_r = p_rad
                convective_switch = 1
            
            # Iterative solution for convective temperature
            # Need iteration because pressure depends on scale height H(T)
            T_old = T_rad * 1.01
            T_new = T_rad
            
            iteration_count = 0
            max_iterations = 100
            
            while (np.abs((T_new - T_old) / T_new) > 1e-4) and (iteration_count < max_iterations):
                current_mu_iter = mu(p_prev)
                #p_conv = p_surf * np.exp(-z / scale_height(T_new, current_mu_iter, g))
                p_conv = p_prev * (1 + delta_p_factor)
                T_conv = T_r * ((p_conv / p_r)**Delta)
                T_old = T_new
                T_new = T_old + 0.1 * (T_conv - T_old)
                iteration_count += 1
            
            # Get opacity at converged state
            current_mix_ratios_conv = mix_ratios_profile(p_conv)
            current_mu_conv = mu(p_conv)
            kr = opacity_calculator.get_rosseland_mean(T_conv, p_conv * 1e6, current_mix_ratios_conv, current_mu_conv)
            #print('convective', kr, T_conv, p_conv, current_mu_conv)
            
            # Update optical depth (tsurf.py does this in convective zones too)
            d_tau = kr * (p_conv - p_prev) * 1e6 / g
            optical_depth = optical_depth + d_tau
            
            T_rad = T_conv
            T_prev = T_conv
            p_prev = p_conv
            
            grad_opt = d_tau / delta_z
        
        # Adaptive step size based on optical depth gradient
        if grad_opt > 1e-8:
            delta_p_factor = 0.01
        else:
            delta_p_factor = 0.1
        
        # Store current state
        z_profile.append(z)
        p_profile.append(p_prev)
        T_profile.append(T_prev)
        tau_profile.append(optical_depth)
        kappa_profile.append(kr)
        Delta_profile.append(Delta)

        i_iter += 1
        if i_iter % 100 == 0:
            print(f'Iteration {i_iter}: P = {p_prev:.3e} bar, T = {T_prev:.3e} K')
    
    # Convert to arrays
    z_profile = np.array(z_profile)
    p_profile = np.array(p_profile)
    T_profile = np.array(T_profile)
    tau_profile = np.array(tau_profile)
    kappa_profile = np.array(kappa_profile)
    Delta_profile = np.array(Delta_profile)
    z = z + z_profile[-1]
    
    # Interpolate to target pressure grid (in log space)
    log_p_profile = np.log10(p_profile)
    log_target_p_grid = np.log10(target_p_grid)
    
    interpolated_T = np.interp(log_target_p_grid, log_p_profile, T_profile)
    interpolated_tau = np.interp(log_target_p_grid, log_p_profile, tau_profile)
    interpolated_kappa = np.interp(log_target_p_grid, log_p_profile, kappa_profile)
    interpolated_z = np.interp(log_target_p_grid, log_p_profile, z_profile)
    interpolated_Delta = np.interp(log_target_p_grid, log_p_profile, Delta_profile)
    
    extra_info = {
        'rosseland_mean': interpolated_kappa,
        'tau': interpolated_tau,
        'z': interpolated_z,
        'T_r': T_r,
        'p_r': p_r,
        'nabla_ad': interpolated_Delta
    }
    
    return interpolated_T, extra_info


