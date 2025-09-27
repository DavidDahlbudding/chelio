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
    new_data = np.zeros((n_layers, len(new_header)))

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
    mu = np.zeros(n_layers)
    a_tot = np.zeros(n_layers)

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
