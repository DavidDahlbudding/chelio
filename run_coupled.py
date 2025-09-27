import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

# Import the refactored modules
from chelio_sim import abundances, external_runners, init_pt, mixfile_utils


def setup_logging(log_dir, config):
    """Configures logging to file and console."""
    log_file = os.path.join(log_dir, config["log_file_name"])
    log_level = getattr(logging, config["level"].upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(log_level)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        #"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        "%(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def expand_env_vars(config):
    """
    Recursively expands environment variables in a config dictionary.
    Any string in the format ${VAR_NAME} will be replaced by the value
    of the environment variable VAR_NAME.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = expand_env_vars(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = expand_env_vars(item)
    elif isinstance(config, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
        return pattern.sub(lambda m: os.environ.get(m.group(1), ""), config)
    return config


def main():
    # 1. PARSE ARGUMENTS
    parser = argparse.ArgumentParser(
        description="Run a coupled HELIOS-GGchem simulation."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--name", required=True, help="Unique name for the simulation run."
    )
    parser.add_argument("--out_dir", default="output", help="Root output directory.")

    # Add overrides for key simulation parameters
    sim_params = [
        "toa_pressure",
        "internal_temp",
        "surface_albedo",
        "melt_temp",
        "melt_frac",
        "h_ocean",
        "ctoh_ratio",
        "ntoc_ratio",
        "fO2",
        "stoc_ratio",
        "cltoc_ratio",
    ]
    for param in sim_params:
        parser.add_argument(f"--{param}", type=float, help=f"Override {param} from config.")
    args = parser.parse_args()

    # 2. LOAD AND PROCESS CONFIGURATION
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = expand_env_vars(config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    # Apply command-line overrides to config
    for param in sim_params:
        if getattr(args, param) is not None:
            config["simulation_params"][param] = getattr(args, param)

    # 3. SETUP PATHS AND DIRECTORIES
    chelio_path = config["paths"].get("chelio_path") or Path(__file__).parent.resolve()
    ggchem_path = os.path.expanduser(config["paths"]["ggchem_path"])
    helios_path = os.path.expanduser(config["paths"]["helios_path"])

    if (
        not ggchem_path
        or not os.path.isdir(ggchem_path)
        or not helios_path
        or not os.path.isdir(helios_path)
    ):
        print("Error: Please set GGCHEM_PATH and HELIOS_PATH environment variables,")
        print("       or specify the paths directly in config.yaml.")
        sys.exit(1)

    if args.out_dir[-1] != "/":
        args.out_dir += "/"
    run_output_dir = os.path.join(chelio_path, args.out_dir, args.name)
    run_output_dir_outgassed = os.path.join(chelio_path, args.out_dir, f"{args.name}_outgassed")
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_output_dir_outgassed, exist_ok=True)

    # 4. SETUP LOGGING
    log = setup_logging(run_output_dir, config["logging"])
    log.info(f"--- Starting Chelio Simulation: {args.name} ---")
    log.info(f"Output directory: {run_output_dir}")

    # 5. EXECUTE SIMULATION LOGIC (Mirrors the bash script)
    try:
        sim_p = config["simulation_params"]
        boa_p_threshold = sim_p["boa_pressure_threshold"]

        # --- Initial abundance calculation and HELIOS run for outgassed atmosphere ---
        outgassed_tp_file = os.path.join(run_output_dir_outgassed, f"{args.name}_outgassed_tp.dat")
        if not os.path.exists(outgassed_tp_file):
            abundances.calculate_abundances(
                output_dir="helios",
                melt_frac=sim_p["melt_frac"],
                T_surf=sim_p["melt_temp"],
                H_ocean=sim_p["h_ocean"],
                CtoH=sim_p["ctoh_ratio"],
                NtoC=sim_p["ntoc_ratio"],
                fO2=sim_p["fO2"],
                StoC=sim_p.get("stoc_ratio"),
                CltoC=sim_p.get("cltoc_ratio"),
            )
            shutil.copy(os.path.join(chelio_path, "helios_inputs", "species.dat"), os.path.join(run_output_dir_outgassed, "species.dat"))
            p_boa_path = os.path.join(chelio_path, "helios_inputs", "P_BOA.dat")
            shutil.copy(p_boa_path, os.path.join(run_output_dir_outgassed, "P_BOA.dat"))

            with open(p_boa_path, "r") as f:
                boa_p = float(f.read().strip())
            
            if boa_p < float(boa_p_threshold):
                log.error(f"P_BOA ({boa_p}) is less than {boa_p_threshold} dyn/cm^2. Exiting...")
                sys.exit(1)

            log.info("Running HELIOS with outgassed molecular species...")
            helios_params_outgas = {
                "name": f"{args.name}_outgassed",
                "output_directory": os.path.join(chelio_path, args.out_dir),
                "toa_pressure": sim_p["toa_pressure"],
                "boa_pressure": boa_p,
                "internal_temperature": sim_p["internal_temp"],
                "surface_albedo": sim_p["surface_albedo"],
                "path_to_species_file": os.path.join(chelio_path, "helios_inputs", "species.dat"),
                "coupling_mode": "no",
                "coupling_iteration_step": 0,
                "coupling_speed_up": "no",
                "write_tp_profile_during_run": 30000,
                "maximum_number_of_iterations": 30001,
                "radiative_equilibrium_criterion": config["coupling"]["rad_eq_criterion"],
            }
            external_runners.run_helios(helios_path, helios_params_outgas)
        else:
            log.info(f"Skipping initial outgassed HELIOS run as {outgassed_tp_file} already exists.")

        if not config["coupling"]["with_ggchem"]:
            log.info("`with_ggchem` is False. Exiting after initial HELIOS run.")
            sys.exit(0)

        # --- Initial GGchem Setup and Run ---
        log.info("Initializing GGchem with initial abundances and P-T profile...")
        abundances.calculate_abundances(
            output_dir="ggchem",
            melt_frac=sim_p["melt_frac"],
            T_surf=sim_p["melt_temp"],
            H_ocean=sim_p["h_ocean"],
            CtoH=sim_p["ctoh_ratio"],
            NtoC=sim_p["ntoc_ratio"],
            fO2=sim_p["fO2"],
            StoC=sim_p.get("stoc_ratio"),
            CltoC=sim_p.get("cltoc_ratio"),
        )
        p_boa_path = os.path.join(chelio_path, "helios_inputs", "P_BOA.dat")
        with open(p_boa_path, "r") as f:
            boa_p = float(f.read().strip())

        if boa_p < float(boa_p_threshold):
            log.error(f"P_BOA ({boa_p}) is less than {boa_p_threshold} dyn/cm^2. Exiting...")
            sys.exit(1)
        
        shutil.copy(os.path.join(chelio_path, 'helios_inputs', 'species.dat'), run_output_dir)
        shutil.copy(p_boa_path, os.path.join(run_output_dir, "P_BOA.dat"))

        init_pt.create_pt_profile(Teq=500, Pmin=sim_p["toa_pressure"], Pmax=boa_p)

        # Prepare GGchem's working directory
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), os.path.join(ggchem_path, 'abund_helios.in'))
        ggchem_pt_input = os.path.join(ggchem_path, 'structures', 'pt_helios.in')
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), ggchem_pt_input)
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'param.in'), os.path.join(ggchem_path, 'input', 'param_helios.in'))
        
        # Archive initial inputs
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), run_output_dir)
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), os.path.join(run_output_dir, f"{args.name}_tp_coupling_-1.dat"))

        log.info("Running initial GGchem calculation...")
        external_runners.run_ggchem(ggchem_path)
        
        # --- Coupling Loop ---
        i_min = config["coupling"]["i_min"]
        i_max = config["coupling"]["i_max"]
        i_full = config["coupling"]["i_full_convergence"]
        
        started_convection = 0
        coupling_speed_up = "no"

        for i in range(i_min, i_max + 1):
            log.info(f"--- Coupling Iteration: {i} ---")

            # Convert GGchem output to HELIOS mixfile
            ggchem_output = os.path.join(ggchem_path, "Static_Conc.dat")
            helios_mixfile = os.path.join(run_output_dir, f"vertical_mix_{i}.dat")
            mixfile_utils.convert_ggchem_to_helios(ggchem_output, helios_mixfile)
            shutil.copy(ggchem_output, os.path.join(run_output_dir, f"Static_Conc_{i}.dat"))

            # Determine HELIOS parameters for this iteration
            if i == 0:
                max_iter = config["coupling"]["helios_max_iter_initial"]
            elif i >= i_full:
                max_iter = config["coupling"]["helios_max_iter_full"]
                coupling_speed_up = "yes"
            else:
                max_iter = config["coupling"]["helios_max_iter_intermediate"]

            # Check for previous convection status
            convection_file = os.path.join(run_output_dir, f"{args.name}_started_convection.dat")
            if os.path.exists(convection_file):
                with open(convection_file, 'r') as f:
                    started_convection = int(f.read().strip())
                log.info(f"Previous convection status: {started_convection}")

            # Prepare HELIOS params
            helios_params = {
                "name": args.name,
                "output_directory": os.path.join(chelio_path, args.out_dir),
                "toa_pressure": sim_p["toa_pressure"],
                "boa_pressure": boa_p,
                "internal_temperature": sim_p["internal_temp"],
                "surface_albedo": sim_p["surface_albedo"],
                "path_to_species_file": os.path.join(chelio_path, "helios_inputs", "species.dat"),
                "file_with_vertical_mixing_ratios": helios_mixfile,
                "path_to_temperature_file": os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i-1}.dat"),
                "coupling_mode": "yes",
                "coupling_iteration_step": i,
                "coupling_speed_up": coupling_speed_up,
                "started_convection": started_convection,
                "write_tp_profile_during_run": max_iter,
                "maximum_number_of_iterations": max_iter + 1,
                "radiative_equilibrium_criterion": config["coupling"]["rad_eq_criterion"],
            }
            external_runners.run_helios(helios_path, helios_params)

            # Check for convergence
            convergence_file = os.path.join(run_output_dir, f"{args.name}_coupling_convergence.dat")
            if os.path.exists(convergence_file):
                with open(convergence_file, "r") as f:
                    stop_flag = int(f.read().strip())
                if stop_flag == 1:
                    log.info("Coupling converged. Stopping iterations.")
                    break

            # Prepare for next GGchem run
            new_tp_profile = os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i}.dat")
            shutil.copy(new_tp_profile, ggchem_pt_input)

            # Run GGchem
            external_runners.run_ggchem(ggchem_path)

        log.info(f"--- Finalizing Simulation ---")
        # Final conversion of GGchem output
        final_mixfile = os.path.join(run_output_dir, f"vertical_mix_{i+1}.dat")
        mixfile_utils.convert_ggchem_to_helios(ggchem_output, final_mixfile)
        shutil.copy(ggchem_output, os.path.join(run_output_dir, f"Static_Conc_{i+1}.dat"))

        log.info(f"Simulation '{args.name}' completed.")

    except Exception:
        log.critical("An unhandled error occurred during the simulation.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
