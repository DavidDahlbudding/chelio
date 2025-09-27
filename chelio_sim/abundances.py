import argparse
import logging
import os
import sys

import numpy as np

# This import is dependent on the HELIOS_PATH environment variable.
# The main run_coupled.py script will ensure this is set.
try:
    helios_path = os.environ["HELIOS_PATH"]
    if helios_path not in sys.path:
        sys.path.append(helios_path)
    from source.species_database import species_lib
except (ImportError, KeyError):
    # This will be logged by the function, but we need a placeholder if the script is run directly
    # without the environment variable set.
    species_lib = None

# It is assumed that atmodeller is installed in the environment.
from atmodeller import (
    InteriorAtmosphere,
    Planet,
    Species,
    SpeciesCollection,
    earth_oceans_to_hydrogen_mass,
)
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

log = logging.getLogger(__name__)

M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m


def calculate_abundances(
    output_dir,
    melt_frac,
    T_surf,
    H_ocean,
    CtoH,
    NtoC,
    fO2,
    M_P=M_EARTH,
    R_P=R_EARTH,
    core_frac=0.295334691460966,
    StoC=None,
    CltoC=None,
):
    """
    Calculates atmospheric abundances using atmodeller based on interior-atmosphere exchange.

    This function sets up a planet, defines chemical species, and solves for the
    resulting atmospheric composition given various constraints. The results are
    then formatted for use in either GGchem or HELIOS.
    """
    log.info(f"Calculating abundances for '{output_dir}' output...")

    if species_lib is None:
        log.error("Could not import species_database from HELIOS. Is HELIOS_PATH set?")
        raise ImportError("species_database not found.")

    if output_dir not in ["ggchem", "helios"]:
        raise ValueError("Output must be either ggchem or helios")

    # Setup Planet
    planet = Planet(
        planet_mass=M_P,
        mantle_melt_fraction=melt_frac,
        surface_radius=R_P,
        surface_temperature=T_surf,
        core_mass_fraction=core_frac,
    )

    solubility_models = get_solubility_models()

    # Default model only includes C-H-N-O
    H2O_g: Species = Species.create_gas(
        "H2O", solubility=solubility_models["H2O_basalt_dixon95"]
    )
    H2_g: Species = Species.create_gas(
        "H2", solubility=solubility_models["H2_basalt_hirschmann12"]
    )
    CO_g: Species = Species.create_gas(
        "CO", solubility=solubility_models["CO_basalt_yoshioka19"]
    )
    CO2_g: Species = Species.create_gas(
        "CO2", solubility=solubility_models["CO2_basalt_dixon95"]
    )
    CH4_g: Species = Species.create_gas(
        "CH4", solubility=solubility_models["CH4_basalt_ardia13"]
    )
    N2_g: Species = Species.create_gas(
        "N2", solubility=solubility_models["N2_basalt_libourel03"]
    )
    O2_g: Species = Species.create_gas("O2")
    NH3_g: Species = Species.create_gas("NH3")
    C_s = Species.create_condensed("C")  # condensed graphite

    species = (H2O_g, H2_g, CO_g, CO2_g, CH4_g, N2_g, O2_g, NH3_g, C_s)

    h_kg = earth_oceans_to_hydrogen_mass(H_ocean)
    c_kg = CtoH * h_kg
    n_kg = NtoC * c_kg
    mass_constraints = {"H": h_kg, "C": c_kg, "N": n_kg}
    fugacity_constraints = {O2_g.name: IronWustiteBuffer(fO2)}

    if StoC is not None:
        S2_g: Species = Species.create_gas(
            "S2", solubility=solubility_models["S2_basalt_boulliung23"]
        )
        SO2_g: Species = Species.create_gas("SO2")
        H2S_g: Species = Species.create_gas("H2S")
        SO_g: Species = Species.create_gas("SO")
        species += (S2_g, SO2_g, H2S_g, SO_g)
        mass_constraints["S"] = StoC * c_kg

    if CltoC is not None:
        Cl2_g: Species = Species.create_gas(
            "Cl2", solubility=solubility_models["Cl2_basalt_thomas21"]
        )
        ClH_g: Species = Species.create_gas("ClH")
        species += (Cl2_g, ClH_g)
        mass_constraints["Cl"] = CltoC * c_kg

    species = SpeciesCollection(species)
    interior_atmosphere = InteriorAtmosphere(species)

    # Initial solution guess number density (molecules/m^3)
    initial_log_number_density = 50 * np.ones(len(species))

    interior_atmosphere.solve(
        planet=planet,
        initial_log_number_density=initial_log_number_density,
        mass_constraints=mass_constraints,
        fugacity_constraints=fugacity_constraints,
    )

    output = interior_atmosphere.output
    output_dict = output.asdict()

    log.info("Saving abundance calculation results...")

    # --- Write GGchem output if requested ---
    if output_dir == "ggchem":
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../ggchem_inputs/abundances.in")
        )
        x_H = (
            output_dict["element_H"]["atmosphere_moles"]
            / output_dict["atmosphere"]["element_moles"]
        )[0]
        with open(filename, "w") as f:
            for e in list(mass_constraints.keys()) + ["O"]:
                x_e = (
                    output_dict[f"element_{e}"]["atmosphere_moles"]
                    / output_dict["atmosphere"]["element_moles"]
                )[0]
                x_e = np.log10(x_e / x_H) + 12
                log.debug(f"{e} {x_e:.5f}")
                f.write(f"{e} {x_e:.5f}\n")

    # --- Write HELIOS species and P_BOA files ---
    # These files are needed by both GGchem and HELIOS workflows.

    all_helios_species_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../helios_inputs/all_species.dat")
    )
    translate_dict = {"H3N": "NH3", "O2S": "SO2", "OS": "SO"}
    outgassed_species_vmrs = {}
    for key in output_dict.keys():
        if key.endswith("_g"):
            vmr = output_dict[key]["volume_mixing_ratio"][0]
            species_name = key[:-2]
            outgassed_species_vmrs[
                translate_dict.get(species_name, species_name)
            ] = vmr

    species_output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../helios_inputs/species.dat")
    )
    with open(species_output_path, "w") as f_out, open(
        all_helios_species_path, "r"
    ) as f_all:
        for i, line in enumerate(f_all):
            line = line.strip()
            if i == 0:
                log.debug(line)
                f_out.write(line + "\n\n")
            elif line:
                parts = line.split()
                species_name = parts[0]
                is_cia = species_name.startswith("CIA")

                if is_cia:
                    pair = species_lib[species_name].fc_name.replace("1", "").split("&")
                    if pair[0] in outgassed_species_vmrs and pair[1] in outgassed_species_vmrs:
                        if output_dir == "helios":
                            vmr_line = f"{outgassed_species_vmrs[pair[0]]:.5e}&{outgassed_species_vmrs[pair[1]]:.5e}"
                            line = line.replace("file", vmr_line)
                        f_out.write(line + "\n\n")
                        log.debug(line)
                elif species_name in outgassed_species_vmrs:
                    if output_dir == "helios":
                        line = line.replace(
                            "file", f"{outgassed_species_vmrs[species_name]:.5e}"
                        )
                    f_out.write(line + "\n\n")
                    log.debug(line)

    p_boa_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../helios_inputs/P_BOA.dat")
    )
    with open(p_boa_path, "w") as f:
        # Convert from bar to dyn/cm^2 for HELIOS
        boa_p = output_dict["atmosphere"]["pressure"][0] * 1e6
        f.write(f"{boa_p:.5e}")
    log.info(f"Wrote P_BOA = {boa_p:.5e} dyn/cm^2 to {p_boa_path}")


if __name__ == "__main__":
    # This block allows for standalone testing.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Calculate atmospheric abundances.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ggchem",
        help="Output format ('ggchem' or 'helios')",
    )
    parser.add_argument(
        "--melt_frac", type=float, default=1.0, help="Melt fraction of the mantle"
    )
    parser.add_argument(
        "--T_surf", type=float, default=2000.0, help="Surface temperature (K)"
    )
    parser.add_argument(
        "--H_ocean",
        type=float,
        default=1.0,
        help="H content in Earth oceans equivalent",
    )
    parser.add_argument("--CtoH", type=float, default=1.0, help="C/H mass ratio")
    parser.add_argument("--NtoC", type=float, default=0.1, help="N/C mass ratio")
    parser.add_argument(
        "--fO2", type=float, default=0.0, help="Oxygen fugacity fO2 [delta IW]"
    )
    parser.add_argument("--StoC", type=float, default=None, help="S/C mass ratio")
    parser.add_argument("--CltoC", type=float, default=None, help="Cl/C mass ratio")

    args = parser.parse_args()

    if "HELIOS_PATH" not in os.environ:
        log.error("Please set the HELIOS_PATH environment variable for testing.")
        sys.exit(1)

    calculate_abundances(
        output_dir=args.output_dir,
        melt_frac=args.melt_frac,
        T_surf=args.T_surf,
        H_ocean=args.H_ocean,
        CtoH=args.CtoH,
        NtoC=args.NtoC,
        fO2=args.fO2,
        StoC=args.StoC,
        CltoC=args.CltoC,
    )
