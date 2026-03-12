import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml
import numpy as np

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
    parser.add_argument("--outgas_or_manual", default="manual", help="Outgassing or manual mode.")
    parser.add_argument("--with_outgassed", default=False, help="Whether to use outgassed atmosphere.")
    parser.add_argument(
        "--chemistry_mode", 
        choices=["ggchem", "constant"], 
        default="ggchem",
        help="Chemistry mode: 'ggchem' for equilibrium chemistry, 'constant' for constant mixing ratios with condensation."
    )
    parser.add_argument(
        "--constant_mixing_ratios",
        type=str,
        default=None,
        help="Constant mixing ratios as comma-separated key=value pairs (e.g., 'N2=0.5,CH4=0.5,CO2=0.0,H2=0.0,H2O=0.0'). Overrides config."
    )
    parser.add_argument(
        "--relative_humidity",
        type=float,
        default=1.0,
        help="Fractional relative humidity cap for H2O in constant chemistry mode (e.g., 0.8 means VMR_H2O <= 0.8 * p_sat(T)/P). Default: 1.0."
    )

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
        "min_boa_pressure",
        "max_boa_pressure",
        "surface_pressure",
        "a_h",
        "a_c",
        "a_o",
        "a_n",
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
    
    # Apply chemistry_mode override
    if args.chemistry_mode is not None:
        config["coupling"]["chemistry_mode"] = args.chemistry_mode
    # Default to ggchem if not specified
    if "chemistry_mode" not in config.get("coupling", {}):
        config.setdefault("coupling", {})["chemistry_mode"] = "ggchem"

    # Apply constant_mixing_ratios override from command line
    if args.constant_mixing_ratios is not None:
        # Parse comma-separated key=value pairs: "N2=0.5,CH4=0.5,CO2=0.0"
        mixing_ratios = {}
        for pair in args.constant_mixing_ratios.split(","):
            key, value = pair.strip().split("=")
            mixing_ratios[key.strip()] = float(value.strip())
        config["simulation_params"]["constant_mixing_ratios"] = mixing_ratios

    relative_humidity = args.relative_humidity

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

    # Extract iteration parameters
    i_min = config["coupling"]["i_min"]
    i_max = config["coupling"]["i_max"]
    i_full = config["coupling"]["i_full_convergence"]

    # Check if run already exists if {name}_coupling_convergence.dat exists and contains "1" or if {name}_tp_coupling_{i_max}.dat exists
    convergence_file = os.path.join(run_output_dir, f"{args.name}_coupling_convergence.dat")
    final_tp_file = os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i_max}.dat")
    if (os.path.exists(convergence_file) and open(convergence_file).read().strip() == "1") or os.path.exists(final_tp_file):
        print(f"Error: A completed run with name '{args.name}' already exists in {run_output_dir}.")
        print("       Please choose a different name or remove the existing run.")
        sys.exit(1)

    # 4. SETUP LOGGING
    log = setup_logging(run_output_dir, config["logging"])
    log.info(f"--- Starting Chelio Simulation: {args.name} ---")
    log.info(f"Output directory: {run_output_dir}")

    shutil.copy(os.path.join(chelio_path, 'helios_inputs', 'param.dat'), os.path.join(helios_path, 'param.dat'))

    # 5. EXECUTE SIMULATION LOGIC
    try:
        sim_p = config["simulation_params"]
        min_boa_pressure = sim_p["min_boa_pressure"]
        max_boa_pressure = sim_p["max_boa_pressure"]

        # --- Initial abundance calculation and HELIOS run for outgassed atmosphere ---
        outgassed_tp_file = os.path.join(run_output_dir_outgassed, f"{args.name}_outgassed_tp.dat")
        if sim_p["outgas_or_manual"] == "outgas" and not os.path.exists(outgassed_tp_file) and sim_p["with_outgassed"]:
            
            os.makedirs(run_output_dir_outgassed, exist_ok=True)

            abundances.calculate_abundances(
                output_dir="helios",
                melt_frac=sim_p["melt_frac"],
                T_surf=sim_p["melt_temp"],
                H_ocean=sim_p["h_ocean"],
                CtoH=sim_p["ctoh_ratio"],
                NtoC=sim_p["ntoc_ratio"],
                fO2=sim_p["fO2"],
                StoC=sim_p["stoc_ratio"],
                CltoC=sim_p["cltoc_ratio"],
            )
            shutil.copy(os.path.join(chelio_path, "helios_inputs", "species.dat"), os.path.join(run_output_dir_outgassed, "species.dat"))
            p_boa_path = os.path.join(chelio_path, "helios_inputs", "P_BOA.dat")
            shutil.copy(p_boa_path, os.path.join(run_output_dir_outgassed, "P_BOA.dat"))

            with open(p_boa_path, "r") as f:
                boa_p = float(f.read().strip())
            
            if boa_p < float(min_boa_pressure):
                log.error(f"P_BOA ({boa_p}) is less than {min_boa_pressure} dyn/cm^2. Exiting...")
                sys.exit(1)
            elif boa_p > float(max_boa_pressure):
                log.error(f"P_BOA ({boa_p}) is greater than {max_boa_pressure} dyn/cm^2. Exiting...")
                sys.exit(1)

            log.info(f"P_BOA ({boa_p}) is within the range of {min_boa_pressure} to {max_boa_pressure} dyn/cm^2.")

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
            log.info(f"Skipping initial outgassed HELIOS run as {outgassed_tp_file} already exists or outgassing is disabled.")

        if not config["coupling"]["with_ggchem"] and config["coupling"]["chemistry_mode"] == "ggchem":
            log.info("`with_ggchem` is False. Exiting after initial HELIOS run.")
            sys.exit(0)

        # --- Determine chemistry mode ---
        chemistry_mode = config["coupling"]["chemistry_mode"]
        log.info(f"Chemistry mode: {chemistry_mode}")

        # --- Initial Setup ---
        if chemistry_mode == "ggchem":
            log.info("Initializing GGchem with initial abundances and P-T profile...")
            if sim_p["outgas_or_manual"] == "outgas":
                abundances.calculate_abundances_atmodeller(
                    output_dir="ggchem",
                    melt_frac=sim_p["melt_frac"],
                    T_surf=sim_p["melt_temp"],
                    H_ocean=sim_p["h_ocean"],
                    CtoH=sim_p["ctoh_ratio"],
                    NtoC=sim_p["ntoc_ratio"],
                    fO2=sim_p["fO2"],
                    StoC=sim_p["stoc_ratio"],
                    CltoC=sim_p["cltoc_ratio"],
                )
                p_boa_path = os.path.join(chelio_path, "helios_inputs", "P_BOA.dat")
                with open(p_boa_path, "r") as f:
                    boa_p = float(f.read().strip())

                if boa_p < float(min_boa_pressure):
                    log.error(f"P_BOA ({boa_p}) is less than {min_boa_pressure} dyn/cm^2. Exiting...")
                    sys.exit(1)
                elif boa_p > float(max_boa_pressure):
                    log.error(f"P_BOA ({boa_p}) is greater than {max_boa_pressure} dyn/cm^2. Exiting...")
                    sys.exit(1)

                log.info(f"P_BOA ({boa_p}) is within the range of {min_boa_pressure} to {max_boa_pressure} dyn/cm^2.")
                
                shutil.copy(os.path.join(chelio_path, 'helios_inputs', 'species.dat'), run_output_dir)
                shutil.copy(p_boa_path, os.path.join(run_output_dir, "P_BOA.dat"))
            else:
                abundances.calculate_abundances_manual(
                    output_dir="ggchem",
                    a_H=sim_p["a_h"],
                    a_C=sim_p["a_c"],
                    a_O=sim_p["a_o"],
                    a_N=sim_p["a_n"],
                )
                boa_p = float(sim_p["surface_pressure"])

            if i_min == 0:
                P_bar, T_k = init_pt.create_pt_profile(Teq=500, Pmin=float(sim_p["toa_pressure"]), Pmax=boa_p, return_data=True)
            else:
                try:
                    # copy f"{args.name}_tp_coupling_{i_min-1}.dat" to ggchem_inputs/pt_helios.in for the initial run
                    shutil.copy(os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i_min-1}.dat"), os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'))

                    tp_data = np.loadtxt(os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i_min-1}.dat"), skiprows=1)
                    P_bar = tp_data[:, 0]
                    T_k = tp_data[:, 1]
                except FileNotFoundError:
                    log.error(f"Initial T-P profile file not found: {os.path.join(run_output_dir, f'{args.name}_tp_coupling_{i_min-1}.dat')}")
                    sys.exit(1)

            # Prepare GGchem's working directory
            shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), os.path.join(ggchem_path, 'abund_helios.in'))
            ggchem_pt_input = os.path.join(ggchem_path, 'structures', 'pt_helios.in')
            shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), ggchem_pt_input)
            shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'param.in'), os.path.join(ggchem_path, 'input', 'param_helios.in'))
            
            # Archive initial inputs
            shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), run_output_dir)
            if i_min == 0:
                shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), os.path.join(run_output_dir, f"{args.name}_tp_coupling_-1.dat"))

            log.info("Running initial GGchem calculation...")
            external_runners.run_ggchem(ggchem_path)
        
        elif chemistry_mode == "constant":
            log.info("Using constant mixing ratios with condensation limits...")
            
            # Get constant mixing ratios from config
            if "constant_mixing_ratios" not in sim_p:
                log.error("constant_mixing_ratios not found in simulation_params. Please specify mixing ratios in config.")
                sys.exit(1)
            
            constant_mixing_ratios = sim_p["constant_mixing_ratios"]
            log.info(f"Constant mixing ratios: {constant_mixing_ratios}")
            
            # Get surface pressure
            boa_p = float(sim_p["surface_pressure"])
            
            # Create initial P-T profile
            if i_min == 0:
                P_bar, T_k = init_pt.create_pt_profile(Teq=500, Pmin=float(sim_p["toa_pressure"]), Pmax=boa_p, return_data=True)

                # Save initial P-T profile
                initial_tp_path = os.path.join(run_output_dir, f"{args.name}_tp_coupling_-1.dat")
                np.savetxt(initial_tp_path, np.vstack([P_bar, T_k]).T, fmt="%.6e", header="# P [bar], T [K]", comments="")

                # Create initial mixfile with constant mixing ratios
                initial_mixfile = os.path.join(run_output_dir, "vertical_mix_initial.dat")
                mixfile_utils.create_constant_mixfile(P_bar, T_k, constant_mixing_ratios, initial_mixfile, relative_humidity=relative_humidity)
            else:
                try:
                    # copy f"{args.name}_tp_coupling_{i_min-1}.dat" to ggchem_inputs/pt_helios.in for the initial run
                    shutil.copy(os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i_min-1}.dat"), os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'))

                    tp_data = np.loadtxt(os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i_min-1}.dat"), skiprows=1)
                    P_bar = tp_data[:, 0]
                    T_k = tp_data[:, 1]
                    
                except FileNotFoundError:
                    log.error(f"Initial T-P profile file not found: {os.path.join(run_output_dir, f'{args.name}_tp_coupling_{i_min-1}.dat')}")
                    sys.exit(1)
        
        # --- Coupling Loop ---
        
        started_convection = 0
        coupling_speed_up = "no"

        # Initial mixfile creation
        helios_mixfile = os.path.join(run_output_dir, f"vertical_mix_{i_min}.dat")
        
        if chemistry_mode == "ggchem":
            # Convert GGchem output to HELIOS mixfile
            ggchem_output = os.path.join(ggchem_path, "Static_Conc.dat")
            shutil.copy(ggchem_output, os.path.join(run_output_dir, f"Static_Conc_{i_min}.dat"))
            mixfile_utils.convert_ggchem_to_helios(ggchem_output, helios_mixfile)
        elif chemistry_mode == "constant":
            # Create mixfile with constant mixing ratios
            mixfile_utils.create_constant_mixfile(P_bar, T_k, constant_mixing_ratios, helios_mixfile, relative_humidity=relative_humidity)
        
        # copy delad table to run_output_dir and append iteration number to the filename
        delad_table_name = os.path.basename(mixfile_utils.DEFAULT_DELAD_TABLE_PATH)
        delad_table_output = os.path.join(run_output_dir, f"{delad_table_name[:-4]}_{i_min}.dat")
        shutil.copy(mixfile_utils.DEFAULT_DELAD_TABLE_PATH, delad_table_output)

        for i in range(i_min, i_max + 1):
            log.info(f"--- Coupling Iteration: {i} ---")

            # Determine HELIOS parameters for this iteration
            if i == 0:
                max_iter = config["coupling"]["helios_max_iter_initial"]
            elif i >= i_full:
                max_iter = config["coupling"]["helios_max_iter_full"]
                # read _ABORT.dat file to check if we can speed up convergence
                abort_file = os.path.join(run_output_dir, f"{args.name}_ABORT.dat")
                if os.path.exists(abort_file):
                    with open(abort_file, "r") as f:
                        line = f.readline().split(' ')
                        line = line[3][:-1] # exclude ")" to get iteration number
                        if line.isdigit() and int(line) != i-1:
                            # only "speed up" (avg. with previous iteration) if it converged
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
                "kappa_value": "file",
                "kappa_file_path": mixfile_utils.DEFAULT_DELAD_TABLE_PATH,
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

            # Prepare for next iteration
            new_tp_profile = os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i}.dat")
            helios_mixfile = os.path.join(run_output_dir, f"vertical_mix_{i+1}.dat")

            tp_data = np.loadtxt(new_tp_profile, skiprows=1)
            if np.all(tp_data[:, 1] < 1.1):
                log.warning(f"Temperature profile stuck at 1.001 K. Aborting coupling loop...")
                break
            
            if chemistry_mode == "ggchem":
                # Copy new T-P profile to GGchem input and run GGchem
                shutil.copy(new_tp_profile, os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'))
                shutil.copy(new_tp_profile, ggchem_pt_input)

                # Run GGchem
                # in case input is required, input 200 times newline
                input = "\n" * 210
                external_runners.run_ggchem(ggchem_path, input=input)
                
                # Convert GGchem output to HELIOS mixfile
                shutil.copy(ggchem_output, os.path.join(run_output_dir, f"Static_Conc_{i+1}.dat"))
                mixfile_utils.convert_ggchem_to_helios(ggchem_output, helios_mixfile)
            
            elif chemistry_mode == "constant":
                # Read the new T-P profile and create updated mixfile with condensation
                P_bar_new = tp_data[:, 0]
                T_k_new = tp_data[:, 1]
                mixfile_utils.create_constant_mixfile(P_bar_new, T_k_new, constant_mixing_ratios, helios_mixfile, relative_humidity=relative_humidity)

            # copy delad table to run_output_dir and append iteration number to the filename
            delad_table_output = os.path.join(run_output_dir, f"{delad_table_name[:-4]}_{i+1}.dat")
            shutil.copy(mixfile_utils.DEFAULT_DELAD_TABLE_PATH, delad_table_output)

        log.info(f"--- Finalizing Simulation ---")
        # Final conversion/creation of mixfile
        final_mixfile = os.path.join(run_output_dir, f"vertical_mix_{i+1}.dat")
        
        if chemistry_mode == "ggchem":
            mixfile_utils.convert_ggchem_to_helios(ggchem_output, final_mixfile)
            shutil.copy(ggchem_output, os.path.join(run_output_dir, f"Static_Conc_{i+1}.dat"))
            # remove database.dat in ggchem_path
            try:
                os.remove(os.path.join(ggchem_path, "database.dat"))
            except FileNotFoundError:
                pass
        elif chemistry_mode == "constant":
            # Create final mixfile with final T-P profile
            final_tp = os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i}.dat")
            tp_data = np.loadtxt(final_tp, skiprows=1)
            P_bar_final = tp_data[:, 0]
            T_k_final = tp_data[:, 1]
            mixfile_utils.create_constant_mixfile(P_bar_final, T_k_final, constant_mixing_ratios, final_mixfile, relative_humidity=relative_humidity)

        log.info(f"Simulation '{args.name}' completed.")

    except Exception:
        log.critical("An unhandled error occurred during the simulation.", exc_info=True)
        # remove database.dat in ggchem_path (only if using ggchem mode)
        if config.get("coupling", {}).get("chemistry_mode") == "ggchem":
            try:
                os.remove(os.path.join(ggchem_path, "database.dat"))
            except FileNotFoundError:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
