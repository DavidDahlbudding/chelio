import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml
import numpy as np
from scipy.interpolate import interp1d

# Import the refactored modules
from chelio_sim import abundances, external_runners, init_pt, mixfile_utils, rt_utils
from chelio_sim.rt_utils import OpacityCalculator, calculate_tp_profile, parse_mixfile


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
        description="Run a coupled FastRT-GGchem simulation."
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
    os.makedirs(run_output_dir, exist_ok=True)

    # 4. SETUP LOGGING
    log = setup_logging(run_output_dir, config["logging"])
    log.info(f"--- Starting Chelio Simulation: {args.name} ---")
    log.info(f"Output directory: {run_output_dir}")

    # 5. EXECUTE SIMULATION LOGIC (Mirrors the bash script)
    try:
        sim_p = config["simulation_params"]
        min_boa_pressure = sim_p["min_boa_pressure"]
        max_boa_pressure = sim_p["max_boa_pressure"]

        # --- Initial abundance calculation and HELIOS run for outgassed atmosphere ---
        run_output_dir_outgassed = os.path.join(chelio_path, args.out_dir, f"{args.name}_outgassed")
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
                StoC=sim_p.get("stoc_ratio"),
                CltoC=sim_p.get("cltoc_ratio"),
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

        if not config["coupling"]["with_ggchem"]:
            log.info("`with_ggchem` is False. Exiting after initial HELIOS run.")
            sys.exit(0)

        # --- Initial GGchem Setup and Run ---
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

        P_bar, T_k = init_pt.create_pt_profile(Teq=500, Pmin=float(sim_p["toa_pressure"]), Pmax=boa_p, return_data=True)

        # Prepare GGchem's working directory
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), os.path.join(ggchem_path, 'abund_helios.in'))
        ggchem_pt_input = os.path.join(ggchem_path, 'structures', 'pt_helios.in')
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), ggchem_pt_input)
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'param.in'), os.path.join(ggchem_path, 'input', 'param_helios.in'))
        
        # Archive initial inputs
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'abundances.in'), run_output_dir)
        shutil.copy(os.path.join(chelio_path, 'ggchem_inputs', 'pt_helios.in'), os.path.join(run_output_dir, f"{args.name}_tp_coupling_-1.dat"))

        log.info("Running initial GGchem calculation...")
        input = "\n" * 210
        external_runners.run_ggchem(ggchem_path, input=input)
        
        # --- Initialize Opacity Calculator ---
        # This is a placeholder for getting the species list dynamically
        # For now, we hardcode the species we expect to have opacities for.
        species_for_opacity = rt_utils.read_species_file(os.path.join(chelio_path, 'helios_inputs', 'species.dat'))
        opacity_files = {s: os.path.join(helios_path, "input", "opacity", "r50_kdistr", f"{s}_opac_ip_kdistr.h5") for s in species_for_opacity}
        
        # Check if all opacity files exist
        for s, path in opacity_files.items():
            if not os.path.exists(path):
                log.error(f"Opacity file for {s} not found at {path}. Exiting.")
                sys.exit(1)

        # We need a fine T/P grid for the interpolator, but it's not used yet in the current implementation.
        # Passing placeholder grids.
        opac_calc = OpacityCalculator(species_for_opacity, opacity_files, T_grid=T_k, P_grid=P_bar)
        log.info("Opacity calculator initialized.")

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

            # --- Fast T-P Calculation ---
            p_grid, mu_profile, species, mix_ratios = parse_mixfile(helios_mixfile)

            # Create interpolators for mu and mixing ratios as functions of pressure
            mu_func = interp1d(np.log10(p_grid), mu_profile, bounds_error=False, fill_value="extrapolate")
            mix_ratio_funcs = {
                s: interp1d(np.log10(p_grid), mix_ratios[s], bounds_error=False, fill_value="extrapolate")
                for s in species #if s in opac_calc.species
            }

            def get_mu(p): # p in bar
                return mu_func(np.log10(p))

            def get_mix_ratios(p): # p in bar
                return {s: f(np.log10(p)) for s, f in mix_ratio_funcs.items()}
            
            # Adiabatic gradient (placeholder - should be calculated from thermodynamics)
            # Using a constant value typical for diatomic-dominated gas for now.
            def get_nabla_ad(T, P):
                return 0.285714

            # Calculate T-P profile
            if i > i_min:
                old_T_profile = new_T_profile.copy()
            
            new_T_profile, extra_info = calculate_tp_profile(
                T_eff=sim_p["internal_temp"],
                p_surf=p_grid.max(),
                p_top=p_grid.min(),
                g=float(config["planet_params"]["g"]), 
                mu=get_mu,
                nabla_ad=get_nabla_ad,
                opacity_calculator=opac_calc,
                mix_ratios_profile=get_mix_ratios,
                target_p_grid=p_grid,
                D=2.0
            )

            # Save the new T-P profile in a format GGchem can read
            # The format is simple: two columns, Pressure (dyn/cm^2) and Temperature (K)
            new_tp_profile_path = os.path.join(run_output_dir, f"{args.name}_tp_coupling_{i}.dat")
            tp_data_to_save = np.vstack([p_grid, new_T_profile]).T
            np.savetxt(
                new_tp_profile_path,
                tp_data_to_save,
                fmt="%.6e",
                header="# P [bar], T [K]",
                comments=""
            )
            log.info(f"Saved new T-P profile to {new_tp_profile_path}")

            extra_info_path = os.path.join(run_output_dir, f"extra_info_{i}.dat")
            if extra_info['p_r'] is not None:
                convection = (p_grid > extra_info['p_r']).astype(int)
            else:
                convection = np.zeros_like(p_grid, dtype=int)
            np.savetxt(
                extra_info_path,
                np.vstack([p_grid, extra_info['rosseland_mean'], extra_info['tau'], extra_info['z'], convection]).T,
                fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%d"],
                header="# P [bar], rosseland_mean [cm^2/g], tau, z [cm], convection",
                comments=""
            )
            log.info(f"Saved extra info to {extra_info_path}")

            # Check for convergence (use HELIOS convergence for now)
            # convergence_file = os.path.join(run_output_dir, f"{args.name}_coupling_convergence.dat")
            if i > i_min: # Don't check on the first iteration
                if np.allclose(new_T_profile, old_T_profile, atol=1e-3):
                    log.info("Coupling converged. Stopping iterations.")
                    break

            # Prepare for next GGchem run
            shutil.copy(new_tp_profile_path, ggchem_pt_input)

            # Run GGchem
            input = "\n" * 210
            external_runners.run_ggchem(ggchem_path, input=input)

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
