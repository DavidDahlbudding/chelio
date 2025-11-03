import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# import pandas as pd # No longer needed


# Physical constants in cgs units
H = 6.62607015e-27  # Planck constant (erg*s)
C = 2.99792458e10   # Speed of light (cm/s)
K_B = 1.380649e-16    # Boltzmann constant (erg/K)
AMU = 1.660539e-24    # Atomic mass unit (g)


def read_opac_file(filepath):
    """
    Reads opacity data and grid parameters from an HDF5 file.
    Adapted from HELIOS source.
    """
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
    def __init__(self, species_list, opacity_paths, T_grid, P_grid):
        self.species = species_list
        self.opacity_paths = opacity_paths
        self.T_grid = T_grid
        self.P_grid = P_grid
        self.opac_data = {}
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
            
            # Create interpolator
            points = (ktemp, kpress)
            # This is a placeholder for a more sophisticated interpolation scheme if needed
            # For now, we just store the raw data and will interpolate on the fly
            # A more optimized version would interpolate to a common fine grid here.
            self.opac_data[species] = {
                'interpolator': RegularGridInterpolator(points, opac_k, bounds_error=False, fill_value=None),
                'ktemp': ktemp,
                'kpress': kpress
            }
            
    def get_rosseland_mean(self, T, P, mixing_ratios):
        """
        Calculates the Rosseland mean opacity for a given T, P, and composition.
        """
        total_opac_k = np.zeros((len(self.wave), len(self.gauss_y)))
        
        for species, mix_ratio in mixing_ratios.items():
            if species in self.opac_data and mix_ratio > 0:
                interpolator = self.opac_data[species]['interpolator']
                # The interpolator expects a (2,) point for (T, P)
                opac_values = interpolator([T, P])[0] 
                total_opac_k += mix_ratio * opac_values

        # Avoid division by zero
        total_opac_k[total_opac_k < 1e-100] = 1e-100

        return self._calculate_rosseland_mean(T, total_opac_k)

    def _calculate_rosseland_mean(self, T, opac_k):
        """
        Core calculation of the Rosseland mean opacity.
        """
        expanded_wave = self._expand_wave(self.inter_wave, self.gauss_y)
        dBdT = _dplanck_dT(expanded_wave, T)

        # Numerator of Rosseland mean integral
        num_integrated = np.sum(0.5 * self.delta_wave[:, None] * self.gauss_weight[None, :] * dBdT, axis=1)
        numerator = np.sum(num_integrated)

        # Denominator of Rosseland mean integral
        denom_integrated = np.sum(0.5 * self.delta_wave[:, None] * self.gauss_weight[None, :] * dBdT / opac_k, axis=1)
        denominator = np.sum(denom_integrated)
        
        if denominator == 0:
            return 1e-100 # return a small number if denominator is zero

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
        header_line = f.readline().strip()
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


def calculate_tp_profile(
    T_eff, p_surf, p_top, g, mu, nabla_ad,
    opacity_calculator, mix_ratios_profile, target_p_grid
):
    """
    Calculates the 1D temperature-pressure profile of an atmosphere.

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
        target_p_grid (np.ndarray): The pressure grid for the final output.

    Returns:
        np.ndarray: Interpolated temperature profile on the target_p_grid.
    """
    # Lists to store the calculated profile
    p_profile = []
    T_profile = []

    # Initial conditions at the top of the atmosphere
    p_current = p_top
    T_current = T_eff / (2**0.25)
    optical_depth = 0.0

    p_profile.append(p_current)
    T_profile.append(T_current)

    # Downward integration using a while loop
    while p_current < p_surf:
        p_prev = p_current
        T_prev = T_current
        
        # Get local atmospheric properties
        current_mu = mu(p_prev)
        current_mix_ratios = mix_ratios_profile(p_prev)
        kappa = opacity_calculator.get_rosseland_mean(T_prev, p_prev * 1e6, current_mix_ratios) # P in cgs

        # Define pressure step based on maintaining roughly constant optical depth resolution
        # This is a simple adaptive step-size approach
        dp_cgs = (0.01 * g) / (kappa + 1e-10) # Target d_tau ~ 0.01
        dp_bar = dp_cgs * 1e-6
        
        # Ensure we don't overshoot the surface pressure
        p_current = min(p_prev + dp_bar, p_surf)
        
        # Calculate gradients at the previous layer
        nabla_rad = (3 * kappa * (p_current - p_prev) * 1e6) / (16 * (5.67e-5 / np.pi) * T_prev**4 * g) * T_prev / p_prev

        # Check for convection
        current_nabla_ad = nabla_ad(T_prev, p_prev)
        if nabla_rad > current_nabla_ad: # Convective
            T_current = T_prev * (p_current / p_prev)**current_nabla_ad
        else: # Radiative
            d_tau = kappa * (p_current - p_prev) * 1e6 / g
            optical_depth += d_tau
            T_current = T_eff * (0.5 * (1 + 1.5 * optical_depth))**0.25
            
            # Fallback to convective if temperature decreases with depth (unphysical for pure radiative)
            if T_current < T_prev:
                 T_current = T_prev * (p_current / p_prev)**current_nabla_ad
        
        p_profile.append(p_current)
        T_profile.append(T_current)

        if p_current >= p_surf:
            break

    # Interpolate to the target pressure grid
    interp_func = np.interp
    log_p_profile = np.log10(p_profile)
    log_target_p_grid = np.log10(target_p_grid)
    interpolated_T = interp_func(log_target_p_grid, log_p_profile, T_profile)

    return interpolated_T


