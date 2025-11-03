import logging
import os
import sys
import time

import numpy as np

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

# psat from GGchem
mmHg = 1.3328e+03 # dyn/cm^2
bar = 1.0e+06 # dyn/cm^2
kB = 1.380649e-16 # erg/K

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


def append_profiles(header, data, ref_pt=os.path.join(os.environ["CHELIO_PATH"], "ggchem_inputs", "pt_helios.in")):
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

    missing_data[:,0] = ref_P[i_append:]
    missing_data[:,1] = ref_T[i_append:]

    missing_data[:,2] = (missing_data[:,0] * 1e6) / (kB * missing_data[:,1]) # n_tot (cm^-3)
    # mean molecular weight mu gets calculated later
    missing_data[:,4] = data[-1,4] # electron VMR (usually < 1e-300)

    missing_data[:,5:] = np.minimum(max_vmr, data[-1,5:])

    i_h2 = np.where(header == "H2")[0][0] - 5
    total_vmr = np.sum(missing_data[:,5:], axis=-1)
    while np.any(total_vmr < (1 - 1e-3)):
        mask = missing_data[:,5:] < max_vmr
        mask[:,i_h2] = True # ignore H2 condensation
        
        distribution_factor = missing_data[:,5:]
        distribution_factor[~mask] = 0.0 # don't distribut missing VMR to already saturated species
        distribution_factor = distribution_factor / np.sum(distribution_factor, axis=-1, keepdims=True) # normalize

        missing_data[:,5:] = missing_data[:,5:] + distribution_factor * (1 - total_vmr)[:,np.newaxis]
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


def convert_ggchem_to_helios(ggchem_output_path, helios_mixfile_path):
    """
    Converts GGchem output (Static_Conc.dat) to a HELIOS mixfile.

    Args:
        ggchem_output_path (str): Path to the GGchem output file (e.g., Static_Conc.dat).
        helios_mixfile_path (str): Path to write the output HELIOS mixfile to.
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
        new_data = append_profiles(new_header, new_data)

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
