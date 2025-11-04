#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Define the function that will be executed by each worker process ---
def run_single_simulation(params):
    """
    Takes a tuple of simulation parameters, constructs the command,
    and executes a single run of run_coupled.py.
    """
    # Unpack the parameters tuple
    temp, temp_melt, h_ocean, ctoh, ntoc, fo2, base_out_dir = params

    # Construct a unique name for the run, same as the bash script
    sim_name = (
        f"Earth_Tint={temp}_Tmelt={temp_melt}_"
        f"Hocean={h_ocean}_CtoH={ctoh}_"
        f"NtoC={ntoc}_fO2={fo2}"
    )

    # The main orchestrator script to call
    # This assumes run_grid.py is in the same directory as run_coupled.py
    main_script = "./run_coupled.py"

    # Construct the full command as a list of strings
    command = [
        "python3", main_script,
        "--name", sim_name,
        "--out_dir", base_out_dir,
        # Pass parameters as command-line arguments to override the config
        f"--internal_temp={temp}",
        # Add other parameters as needed. For now, we assume they are in the config.
        # Example: f"--melt_temp={temp_melt}",
    ]

    print(f"Starting simulation: {sim_name}")

    try:
        # We use subprocess.run here. For live output, the 'run_command'
        # function in external_runners.py should be used by run_coupled.py.
        # We capture output here mainly for error reporting in case of failure.
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        # If the run was successful, return its name
        return sim_name, True, ""
    except subprocess.CalledProcessError as e:
        # If the simulation fails, capture the error output
        error_message = (
            f"--- FAILED: {sim_name} ---\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
            "-------------------------"
        )
        return sim_name, False, error_message

# --- Main script execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid of Chelio simulations in parallel."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count(), # Default to the number of available CPUs
        help="Number of simulations to run in parallel. Defaults to all available CPU cores."
    )
    args = parser.parse_args()

    # --- Define Parameter Grid ---
    temps = [50, 100, 150, 200]
    temp_melts = [1000, 2000]
    h_oceans = [0.1, 1.0, 10.0]
    ctohs = [0.1, 1.0, 10.0]
    ntocs = [0.01, 0.1, 1.0]
    fo2s = [-3, 0, 3]
    base_out_dir = "output/test"

    # Use itertools.product to create all combinations
    param_grid = list(itertools.product(temps, temp_melts, h_oceans, ctohs, ntocs, fo2s))
    
    # We need to add the base_out_dir to each parameter tuple for the worker function
    tasks = [( *p, base_out_dir) for p in param_grid]
    total_sims = len(tasks)
    
    print(f"Total number of simulations to run: {total_sims}")
    print(f"Running up to {args.jobs} simulations in parallel.")

    # --- Run Parameter Grid in Parallel ---
    completed_count = 0
    failed_runs = []

    # ProcessPoolExecutor manages a pool of worker processes
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all tasks to the pool. submit() returns a Future object.
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}

        # as_completed() gives us results as soon as they are ready
        for future in tqdm(as_completed(futures), total=total_sims, desc="Running Simulations"):
            sim_name, success, message = future.result()
            completed_count += 1
            if success:
                print(f"({completed_count}/{total_sims}) COMPLETED: {sim_name}")
            else:
                print(f"({completed_count}/{total_sims}) FAILED: {sim_name}. See errors below.")
                print(message)
                failed_runs.append(sim_name)

    # --- Final Summary ---
    print("\n--- Parameter Grid Exploration Complete! ---")
    print(f"Total simulations run: {total_sims}")
    print(f"Successful: {total_sims - len(failed_runs)}")
    print(f"Failed: {len(failed_runs)}")
    if failed_runs:
        print("\nFailed simulation names:")
        for name in failed_runs:
            print(f"- {name}")
