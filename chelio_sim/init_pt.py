import argparse
import logging
import os

import numpy as np

log = logging.getLogger(__name__)


def create_pt_profile(Teq, Pmin, Pmax, output_file=None, return_data=False):
    """
    Creates an initial isothermal Pressure-Temperature (P-T) profile.

    This is used to generate an initial `pt_helios.in` file for GGchem.
    A higher initial temperature can help prevent all species from condensing
    at the start of the simulation.

    Args:
        Teq (float): Equilibrium temperature (K) for the isothermal profile.
        Pmin (float): Minimum pressure at the top of the atmosphere (in 1e-6 bar).
        Pmax (float): Maximum pressure at the bottom of the atmosphere (in 1e-6 bar).
        output_file (str, optional): Path to the output file. Defaults to
                                     '../ggchem_inputs/pt_helios.in' relative to
                                     the script location.
    """
    log.info(
        f"Creating initial P-T profile with Teq={Teq}K, P_range=[{Pmin}, {Pmax}] 1e-6 bar"
    )

    p_min_bar = Pmin * 1e-6
    p_max_bar = Pmax * 1e-6

    # Calculate number of layers required
    nlayer = np.int32(np.ceil(10.5 * np.log10(p_max_bar / p_min_bar)) + 1)

    P_bar = np.logspace(np.log10(p_max_bar), np.log10(p_min_bar), nlayer)
    T_k = np.ones_like(P_bar) * Teq

    if output_file is None:
        output_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../ggchem_inputs/pt_helios.in")
        )

    try:
        with open(output_file, "w") as f:
            f.write("# P [bar], T [K]\n")
            for i in range(nlayer):
                f.write(f"{P_bar[i]:.6e} {T_k[i]:.6e}\n")
        log.info(f"Successfully wrote P-T profile to {output_file}")
    except IOError as e:
        log.error(f"Failed to write P-T profile to {output_file}: {e}")
        raise

    if return_data:
        return P_bar, T_k


if __name__ == "__main__":
    # This block allows for standalone testing.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Create an initial P-T profile.")
    parser.add_argument(
        "--Teq", type=float, default=500, help="Equilibrium Temperature (K)"
    )
    parser.add_argument(
        "--Pmin", type=float, default=1e-6, help="Minimum Pressure (in bar)"
    )
    parser.add_argument(
        "--Pmax", type=float, default=1e0, help="Maximum Pressure (in bar)"
    )
    args = parser.parse_args()

    create_pt_profile(Teq=args.Teq, Pmin=args.Pmin, Pmax=args.Pmax)
