"""
Cloud helpers: the HELIOS Mie-file contract, size-averaged optics, Planck means and the conversion
between cloud optical depth and the HELIOS cloud bottom mixing ratio.

Used by chelio_sim/make_mie_files.py (Mie tables), chelio_sim/run_modes.py (cloud flags),
chelio_sim/grid.py (derive hook `cloud_from_tau`), run_coupled.py and analyze/cloud_test.py.

HELIOS (source/clouds.py) reads one Mie file per particle radius from `path_to_mie_files`:
  - 51 radii r = 10**np.arange(-2, 3.1, 0.1) micron, file names "r{:.6f}.dat".format(r)
  - the path is concatenated with the file name, so it must end in "/"
  - one header line, then 7 columns: lambda [micron], x, C_ext, C_sca, C_abs [cm^2 per particle], w0, g
    (HELIOS reads columns 0, 3, 4 and 6)
  - the same ascending wavelength grid in every file, covering all HELIOS bins: bins outside the
    Mie wavelength range get zero cloud opacity without any warning
"""
import functools
import os

import numpy as np

# Particle radii [micron] hardcoded in HELIOS; the same numpy expression, so the file names match
HELIOS_RADII = 10 ** np.arange(-2, 3.1, 0.1)
# HELIOS weights each radius with the lognormal pdf times this bin width [micron]
HELIOS_DELTA_R = HELIOS_RADII * (10**0.05 - 10**-0.05)

M_U = 1.66053906660e-24  # atomic mass unit [g]
# Mean molecular weight above the water condensation level of the H2-dominated test case
# (H2 + CH4, X_C+O = 1e-2, C/O = 0.59, X_N = 1e-4). Used only to turn tau into f_bot.
MU_DEFAULT = 2.12

H_PLANCK = 6.62607015e-27  # erg s
C_LIGHT = 2.99792458e10  # cm/s
K_BOLTZMANN = 1.380649e-16  # erg/K

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz before numpy 2.0


def mie_filename(r):
    """File name HELIOS expects for radius r [micron]."""
    return "r{:.6f}.dat".format(r)


def lognorm_pdf(r, r_mode, sigma):
    """Lognormal number distribution dN/dr with mode r_mode (verbatim from HELIOS source/clouds.py)."""

    r_median = r_mode / np.exp(-np.log(sigma) ** 2)

    norm_factor = 1 / (r * np.log(sigma) * (2 * np.pi) ** 0.5)
    pdf = norm_factor * np.exp(-0.5 * (np.log(r / r_median) / np.log(sigma)) ** 2)

    return pdf


def missing_mie_files(mie_dir):
    """Names of the HELIOS Mie files that are not in mie_dir (all of them if the directory does not exist)."""
    return [mie_filename(r) for r in HELIOS_RADII if not os.path.isfile(os.path.join(mie_dir, mie_filename(r)))]


def read_mie_file(path):
    """Columns of one Mie file: lambda [micron], x, C_ext, C_sca, C_abs [cm^2], w0, g (one header line)."""
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"{path}: expected 7 columns, got shape {data.shape}")
    return data


def mie_wavelength_range(mie_dir):
    """(min, max) wavelength [micron] of the Mie tables (taken from the first radius, as HELIOS does)."""
    lam = read_mie_file(os.path.join(mie_dir, mie_filename(HELIOS_RADII[0])))[:, 0]
    return lam[0], lam[-1]


def helios_interface_range(opacity_dir):
    """
    (min, max) of the "interface wavelengths" [micron] over the opacity files in opacity_dir that HELIOS reads
    on the fly (<species>_opac_ip*.h5; other .h5 files there may be on other grids).
    """
    import glob
    import h5py

    lo, hi = np.inf, -np.inf
    for path in sorted(glob.glob(os.path.join(opacity_dir, "*_opac_ip*.h5"))):
        with h5py.File(path, "r") as f:
            if "interface wavelengths" not in f:
                continue
            lam = f["interface wavelengths"][:] * 1e4  # cm -> micron
        lo, hi = min(lo, lam.min()), max(hi, lam.max())
    if not np.isfinite(lo):
        raise FileNotFoundError(f"no opacity file with 'interface wavelengths' in {opacity_dir}")
    return lo, hi


@functools.lru_cache(maxsize=16)
def _size_averaged_optics_cached(mie_dir, r_mode, sigma):
    pdf = lognorm_pdf(HELIOS_RADII, r_mode, sigma)
    weights = pdf * HELIOS_DELTA_R
    lam = None
    c_abs = c_sca = g_sca = 0.0
    for r, w in zip(HELIOS_RADII, weights):
        data = read_mie_file(os.path.join(mie_dir, mie_filename(r)))
        if lam is None:
            lam = data[:, 0]
        c_abs = c_abs + w * data[:, 4]
        c_sca = c_sca + w * data[:, 3]
        g_sca = g_sca + w * data[:, 6] * data[:, 3]
    g = np.divide(g_sca, c_sca, out=np.zeros_like(c_sca), where=c_sca > 0)
    return lam, c_abs + c_sca, c_sca, g


def size_averaged_optics(mie_dir, r_mode, sigma):
    """
    Size-averaged cloud optics on the Mie wavelength grid, weighted as in HELIOS (lognormal pdf x delta_r on the
    HELIOS radius grid), with g weighted by the scattering cross section (the corrected HELIOS weighting).

    Returns lambda [micron], C_ext, C_sca [cm^2 per particle] and g.
    """
    lam, c_ext, c_sca, g = _size_averaged_optics_cached(os.path.abspath(mie_dir), float(r_mode), float(sigma))
    return lam.copy(), c_ext.copy(), c_sca.copy(), g.copy()


def planck_lambda(lam_um, T):
    """Planck function B_lambda (arbitrary normalisation is fine for means) at wavelength lam_um [micron]."""
    lam = np.asarray(lam_um, dtype=float) * 1e-4
    with np.errstate(over="ignore"):
        return 2 * H_PLANCK * C_LIGHT**2 / lam**5 / np.expm1(H_PLANCK * C_LIGHT / (lam * K_BOLTZMANN * T))


def planck_mean(lam_um, C, T, dlam_um=None):
    """
    Planck mean of C(lambda) at temperature T over the given wavelengths [micron].
    Without dlam_um the integral is a trapezoid over lam_um (fine grids, e.g. the Mie grid); with dlam_um the
    values are bin averages and weighted with B_lambda(centre) * bin width (e.g. HELIOS bins).
    """
    lam_um = np.asarray(lam_um, dtype=float)
    C = np.asarray(C, dtype=float)
    B = planck_lambda(lam_um, T)
    if dlam_um is None:
        return _trapezoid(C * B, lam_um) / _trapezoid(B, lam_um)
    w = B * np.asarray(dlam_um, dtype=float)
    return np.sum(C * w) / np.sum(w)


def f_bot_for_tau(tau, p_bot, h, c_mean, mu=MU_DEFAULT, g=981.0):
    """
    Cloud bottom mixing ratio (particles per gas molecule) of a HELIOS `manual` deck with total optical depth tau.

    HELIOS sets f = f_bot (p / p_bot)^(1/h - 1) above the cloud base, so the particle column above the base is
    f_bot p_bot h / (mu m_u g) and tau = f_bot p_bot h c_mean / (mu m_u g).
    p_bot [dyn/cm^2], c_mean [cm^2 per particle], g [cm/s^2].
    """
    return tau * mu * M_U * g / (p_bot * h * c_mean)


def tau_for_f_bot(f_bot, p_bot, h, c_mean, mu=MU_DEFAULT, g=981.0):
    """Inverse of f_bot_for_tau."""
    return f_bot * p_bot * h * c_mean / (mu * M_U * g)


def helios_param_value(param_file, label):
    """Value (first token after '=') of the first line of a HELIOS param file that contains `label`, or None."""
    with open(param_file) as f:
        for line in f:
            if label in line and "=" in line:
                tokens = line.split("=", 1)[1].split()
                return tokens[0] if tokens else None
    return None
