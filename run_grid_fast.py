#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# --- Define the function that will be executed by each worker process ---
def run_single_simulation(params):
    """
    Takes a tuple of simulation parameters, constructs the command,
    and executes a single run of run_coupled.py.
    """
    # Unpack the parameters tuple
    temp, psurf, a_h, a_c, a_o, a_n, base_out_dir = params

    psurf_bar = int(psurf/1e6)
    #psurf_string = f"{psurf:.0e}".replace('0', '').replace('+', '')

    # CplusO = a_c + a_o / (a_h + a_c + a_o)
    # CplusO_string = f"{CplusO:.0e}".replace('0', '')
    
    # CtoO = a_c / a_o
    # CtoO = round(CtoO, 2)
    # CtoO_string = f"{CtoO:.2f}"
    # if CtoO_string[-1] == '0':
    #     CtoO_string = CtoO_string[:-1]

    # Construct a unique name for the run, same as the bash script
    sim_name = (
        f"Earth_Tint={temp}K_Psurf={psurf_bar}bar_aH={a_h}_aC={a_c}_aO={a_o}_aN={a_n}"
        #f"Earth_P0={psurf_string}_Tint={temp:.0f}_CplusO={CplusO_string}_CtoO={CtoO_string}"
    )

    # The main orchestrator script to call
    # This assumes run_grid.py is in the same directory as run_fast.py
    main_script = "./run_fast.py"

    # Construct the full command as a list of strings
    command = [
        "python3", main_script,
        "--name", sim_name,
        "--out_dir", base_out_dir,
        # Pass parameters as command-line arguments to override the config
        "--internal_temp", str(temp),
        "--surface_pressure", str(psurf),
        "--a_h", str(a_h),
        "--a_c", str(a_c),
        "--a_o", str(a_o),
        "--a_n", str(a_n),
        "--outgas_or_manual", "manual"
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
        default=1, #os.cpu_count(),
        help="Number of simulations to run in parallel. Defaults to all available CPU cores."
    )
    args = parser.parse_args()

    # --- Define Parameter Grid ---
    temps = [50, 100, 150]
    psurfs = [1.0e8, 1.0e7, 1.0e6]
    a_hs = [1.0e-06, 1.0e-02, 1.0e-01, 1.0e+00, 3.0e+00]
    a_cs = [1.0e-06, 1.0e-02, 1.0e-01, 2.5e-01, 5.0e-01]
    a_os = [1.0e-06, 1.0e-02, 1.0e-01, 5.0e-01, 1.0e+00]
    a_ns = [1.0e+00]

    # CplusOs = np.array([1e-3, 1e-2, 1e-1])
    # CtoOs = np.array([0.1, 0.59, 1.0])

    base_out_dir = "output/N2dom_fast"

    # Use itertools.product to create all combinations
    param_grid = list(itertools.product(temps, psurfs, a_hs, a_cs, a_os, a_ns))
    param_grid_len = len(param_grid)

    # convert to a_h, a_c, a_o, a_n
    # a_cs = CplusOs[:,np.newaxis] * CtoOs[np.newaxis,:] / (1 + CtoOs[np.newaxis,:])
    # a_os = CplusOs[:,np.newaxis] - a_cs
    # a_hs = np.ones_like(a_cs) - CplusOs[:,np.newaxis]
    # a_ns = np.zeros_like(a_cs)

    # param_grid = []
    # for temp in temps:
    #     for psurf in psurfs:
    #         for i_CplusO in range(len(CplusOs)):
    #             for i_CtoO in range(len(CtoOs)):
    #                 param_grid.append((temp, psurf, a_hs[i_CplusO, i_CtoO], a_cs[i_CplusO, i_CtoO], a_os[i_CplusO, i_CtoO], a_ns[i_CplusO, i_CtoO]))
    
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
