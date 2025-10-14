import argparse
import os
import glob
import shutil
import subprocess
import re
import logging

def setup_logging():
    """Configures basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger()

def main():
    parser = argparse.ArgumentParser(
        description="Run GGchem for the last coupling step in simulation folders and copy the result."
    )
    parser.add_argument(
        "--folder", required=True,
        help="Parent folder containing simulation folders (e.g., 'output/Atmodeller').",
    )
    parser.add_argument(
        "--ggchem_path", required=True, help="Path to the GGchem directory."
    )
    parser.add_argument(
        "--chelio_path", default=".", help="Path to the chelio project root."
    )
    args = parser.parse_args()

    log = setup_logging()

    if not os.path.isdir(args.folder):
        log.error(f"Parent folder not found: {args.folder}")
        return

    if not os.path.isdir(args.ggchem_path):
        log.error(f"GGchem path not found: {args.ggchem_path}")
        return

    simulation_folders = [
        f.path for f in os.scandir(args.folder) if f.is_dir()
    ]

    for sim_folder in simulation_folders:
        log.info(f"--- Processing simulation: {os.path.basename(sim_folder)} ---")
        
        # Find the T-P profile with the highest iteration number
        tp_files = glob.glob(os.path.join(sim_folder, "*_tp_coupling_*.dat"))
        if not tp_files:
            log.warning(f"No T-P profiles found in {sim_folder}. Skipping.")
            continue

        latest_tp_file = None
        latest_i = -1
        for f in tp_files:
            match = re.search(r"_tp_coupling_(-?\d+)\.dat$", f)
            if match:
                i = int(match.group(1))
                if i > latest_i:
                    latest_i = i
                    latest_tp_file = f
        
        if latest_tp_file is None:
            log.warning(f"Could not determine the latest T-P profile in {sim_folder}. Skipping.")
            continue

        log.info(f"Found latest T-P profile: {os.path.basename(latest_tp_file)} (i={latest_i})")

        # Check if corresponding ggchem output exists (Static_Conc_{i+1}.dat)
        ggchem_output_file = os.path.join(sim_folder, f"Static_Conc_{latest_i+1}.dat")
        if os.path.exists(ggchem_output_file):
            log.info(f"GGchem output already exists for {sim_folder}. Skipping.")
            continue

        # Check for abundances.in
        abundances_file = os.path.join(sim_folder, "abundances.in")
        if not os.path.exists(abundances_file):
            log.warning(f"abundances.in not found in {sim_folder}. Skipping.")
            continue

        # Prepare GGchem directory
        try:
            # Copy T-P profile
            ggchem_pt_input = os.path.join(args.ggchem_path, 'structures', 'pt_helios.in')
            os.makedirs(os.path.dirname(ggchem_pt_input), exist_ok=True)
            shutil.copy(latest_tp_file, ggchem_pt_input)

            # Copy abundances
            ggchem_abund_input = os.path.join(args.ggchem_path, 'abund_helios.in')
            shutil.copy(abundances_file, ggchem_abund_input)

            # Copy params file (assuming standard chelio structure)
            param_file_src = os.path.join(args.chelio_path, 'ggchem_inputs', 'param.in')
            param_file_dest = os.path.join(args.ggchem_path, 'input', 'param_helios.in')
            if not os.path.exists(param_file_src):
                log.error(f"GGchem param file not found at {param_file_src}. Cannot run GGchem.")
                continue
            os.makedirs(os.path.dirname(param_file_dest), exist_ok=True)
            shutil.copy(param_file_src, param_file_dest)

        except Exception as e:
            log.error(f"Error preparing GGchem directory for {sim_folder}: {e}")
            continue
        
        # Run GGchem
        log.info("Running GGchem...")
        try:
            # To automate pressing "Enter" for GGchem, we send a series of newlines to its stdin.
            # 300 should be more than enough to handle cases where it's not converging.
            ggchem_input = "\n" * 1000
            process = subprocess.run(
                ["./ggchem", "input/param_helios.in"],
                cwd=args.ggchem_path,
                capture_output=True,
                text=True,
                input=ggchem_input,
            )
            if process.returncode != 0:
                log.warning(f"GGchem failed to converge for {sim_folder} (exit code {process.returncode}).")
                log.warning("GGchem stdout:\n" + process.stdout)
                log.warning("GGchem stderr:\n" + process.stderr)
            else:
                log.info("GGchem completed successfully.")

        except FileNotFoundError:
            log.error(f"'./ggchem' executable not found in {args.ggchem_path}. Aborting.")
            return # Abort if ggchem is not found at all
        except Exception as e:
            log.error(f"An unexpected error occurred while running GGchem: {e}")

        # Copy result back, regardless of convergence
        ggchem_output_file = os.path.join(args.ggchem_path, "Static_Conc.dat")
        if os.path.exists(ggchem_output_file):
            dest_file = os.path.join(sim_folder, f"Static_Conc_{latest_i+1}.dat")
            try:
                shutil.copy(ggchem_output_file, dest_file)
                log.info(f"Copied GGchem output to {dest_file}")
            except Exception as e:
                log.error(f"Failed to copy GGchem output for {sim_folder}: {e}")
        else:
            log.warning(f"GGchem output file (Static_Conc.dat) not found after running.")


if __name__ == "__main__":
    import sys
    main()
