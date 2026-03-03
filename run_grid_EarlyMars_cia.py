#!/usr/bin/env python3
"""
Run a grid of Chelio simulations for an Early Mars scenario,
testing different CIA opacity sources for CO2-H2 and CO2-CH4.

Atmosphere: CO2 background, 80% relative humidity H2O, trace H2 or CH4.
Surface pressure: 2 bar.
HELIOS parameter file: helios_inputs/param_EarlyMars.dat.

CIA sources tested:
  CO2-H2:  CO2-H2_2018, CO2-H2_2024
  CO2-CH4: CO2-CH4_2017, CO2-CH4_2020
"""
import argparse
import itertools
import subprocess
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
HELIOS_PATH = os.environ.get("HELIOS_PATH", os.path.expanduser("~/PhD/HELIOS"))
HITRAN_CIA_DIR = os.path.join(HELIOS_PATH, "input", "opacity", "hitran_cia")
R50_KDISTR_DIR = os.path.join(HELIOS_PATH, "input", "opacity", "r50_kdistr")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELIOS_INPUTS_DIR = os.path.join(SCRIPT_DIR, "helios_inputs")
PARAM_EARLY_MARS = os.path.join(HELIOS_INPUTS_DIR, "param_EarlyMars.dat")
PARAM_DEFAULT = os.path.join(HELIOS_INPUTS_DIR, "param.dat")
PARAM_BACKUP = os.path.join(HELIOS_INPUTS_DIR, "param.dat.bak")

# Default CIA sources (used to restore after testing non-default sources)
DEFAULT_CIA_SOURCES = {
    "CO2-H2":  "CO2-H2_2024",
    "CO2-CH4": "CO2-CH4_2024_main",
}

# CIA sources to test per trace gas species
CIA_SOURCES_BY_GAS = {
    "H2":  ["CO2-H2_2018.h5",  "CO2-H2_2024.h5"],
    "CH4": ["CO2-CH4_2017.h5", "CO2-CH4_2020.h5"],
}

# CIA pair associated with each trace gas
CIA_PAIR_BY_GAS = {
    "H2":  "CO2-H2",
    "CH4": "CO2-CH4",
}

# Trace gas mixing ratio grid [%]
TRACE_GAS_PCTS = [0.5, 1, 2, 3, 5, 10]

# Surface pressure: 2 bar = 2e6 dyn/cm²
PSURF = 2e6

# H2O relative humidity
RELATIVE_HUMIDITY = 0.8


def get_cia_target_name(pair):
    pair_no_dash = pair.replace("-", "")
    return f"CIA_{pair_no_dash}_opac_ip_kdistr.h5"


def copy_cia_to_r50(source_filename, pair):
    source_path = os.path.join(HITRAN_CIA_DIR, source_filename)
    target_name = get_cia_target_name(pair)
    target_path = os.path.join(R50_KDISTR_DIR, target_name)
    print(f"Copying {source_filename} -> {target_name}")
    shutil.copy2(source_path, target_path)


def restore_default_cia(pair):
    if pair not in DEFAULT_CIA_SOURCES:
        print(f"Warning: No default CIA source defined for {pair}")
        return
    source_filename = f"{DEFAULT_CIA_SOURCES[pair]}.h5"
    source_path = os.path.join(HITRAN_CIA_DIR, source_filename)
    target_name = get_cia_target_name(pair)
    target_path = os.path.join(R50_KDISTR_DIR, target_name)
    if os.path.exists(source_path):
        print(f"Restoring default CIA: {source_filename} -> {target_name}")
        shutil.copy2(source_path, target_path)
    else:
        print(f"Warning: Default source {source_filename} not found!")


def build_mixing_ratios(trace_gas, trace_pct):
    """
    CO2 background with trace H2 or CH4 and H2O (to be capped at RELATIVE_HUMIDITY * p_sat/P).
    The non-participating gas of H2/CH4 is set to a negligible floor value.
    """
    trace_vmr = trace_pct / 100.0
    non_trace = "CH4" if trace_gas == "H2" else "H2"
    return f"CO2=1.0,H2O=1.0,{trace_gas}={trace_vmr},{non_trace}=1e-30"


def run_single_simulation(params):
    trace_gas, trace_pct, cia_source_name, base_out_dir = params

    pct_str = f"{trace_pct:g}"
    sim_name = f"EarlyMars_{trace_gas}={pct_str}pct_CIA={cia_source_name}"

    mixing_ratios_str = build_mixing_ratios(trace_gas, trace_pct)

    command = [
        "python3", "./run_coupled.py",
        "--name", sim_name,
        "--out_dir", base_out_dir,
        "--surface_pressure", str(PSURF),
        "--chemistry_mode", "constant",
        "--constant_mixing_ratios", mixing_ratios_str,
        "--relative_humidity", str(RELATIVE_HUMIDITY),
        "--outgas_or_manual", "manual",
        "--internal_temp", "0",
        "--surface_albedo", "0.2",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )
        return sim_name, True, ""
    except subprocess.CalledProcessError as e:
        error_message = (
            f"--- FAILED: {sim_name} ---\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
            "-------------------------"
        )
        return sim_name, False, error_message


def run_simulations_for_cia_source(cia_source_filename, cia_source_name, trace_gas, base_out_dir, n_jobs):
    tasks = [
        (trace_gas, pct, cia_source_name, base_out_dir)
        for pct in TRACE_GAS_PCTS
    ]
    total = len(tasks)

    print(f"\n--- {total} sims for CIA source: {cia_source_name}, trace gas: {trace_gas} ---")
    print(f"    Mixing ratios (%): {TRACE_GAS_PCTS}")

    completed_count = 0
    failed_runs = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=total, desc=f"{cia_source_name}"):
            sim_name, success, message = future.result()
            completed_count += 1
            if success:
                print(f"  ({completed_count}/{total}) COMPLETED: {sim_name}")
            else:
                print(f"  ({completed_count}/{total}) FAILED: {sim_name}")
                print(message)
                failed_runs.append(sim_name)

    return failed_runs


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Early Mars CIA grid (CO2 background, H2O @ 80% RH, trace H2 or CH4)."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of simulations to run in parallel."
    )
    parser.add_argument(
        "--gases",
        nargs="+",
        choices=["H2", "CH4"],
        default=["H2", "CH4"],
        help="Trace gas species to test. Default: both H2 and CH4."
    )
    args = parser.parse_args()

    # --- Restore default CIA sources before starting ---
    print("Restoring default CIA sources before starting...")
    for pair in DEFAULT_CIA_SOURCES:
        restore_default_cia(pair)

    # --- Copy param_EarlyMars.dat -> param.dat (back up original first) ---
    if os.path.exists(PARAM_DEFAULT):
        shutil.copy2(PARAM_DEFAULT, PARAM_BACKUP)
        print(f"Backed up {PARAM_DEFAULT} -> {PARAM_BACKUP}")
    shutil.copy2(PARAM_EARLY_MARS, PARAM_DEFAULT)
    print(f"Copied {PARAM_EARLY_MARS} -> {PARAM_DEFAULT}")

    base_out_dir = "output/EarlyMars_CIA_comparison"

    total_sims = sum(
        len(CIA_SOURCES_BY_GAS[gas]) * len(TRACE_GAS_PCTS) for gas in args.gases
    )
    print(f"\n=== Early Mars CIA Grid ===")
    print(f"Trace gases: {args.gases}")
    print(f"Mixing ratios [%]: {TRACE_GAS_PCTS}")
    print(f"Psurf: {PSURF:.2e} dyn/cm² (2 bar)")
    print(f"H2O relative humidity: {RELATIVE_HUMIDITY}")
    print(f"Total simulations: {total_sims}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Output directory: {base_out_dir}")
    print()

    all_failed_runs = []

    try:
        for gas in args.gases:
            pair = CIA_PAIR_BY_GAS[gas]
            sources = CIA_SOURCES_BY_GAS[gas]

            print(f"\n{'='*60}")
            print(f"Trace gas: {gas}  |  CIA pair: {pair}  |  Sources: {sources}")
            print(f"{'='*60}")

            for source_filename in sources:
                source_name = source_filename.replace(".h5", "")
                copy_cia_to_r50(source_filename, pair)

                failed = run_simulations_for_cia_source(
                    source_filename, source_name, gas, base_out_dir, args.jobs
                )
                all_failed_runs.extend(failed)

                default_source = DEFAULT_CIA_SOURCES.get(pair)
                if default_source and source_name != default_source:
                    restore_default_cia(pair)
                else:
                    print(f"Source {source_name} is the default for {pair}, no restoration needed.")

    finally:
        # --- Restore original param.dat ---
        if os.path.exists(PARAM_BACKUP):
            shutil.copy2(PARAM_BACKUP, PARAM_DEFAULT)
            os.remove(PARAM_BACKUP)
            print(f"\nRestored original {PARAM_DEFAULT}")

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print("=== Early Mars CIA Grid Complete! ===")
    print(f"Total simulations run: {total_sims}")
    print(f"Successful: {total_sims - len(all_failed_runs)}")
    print(f"Failed: {len(all_failed_runs)}")
    if all_failed_runs:
        print("\nFailed simulation names:")
        for name in all_failed_runs:
            print(f"  - {name}")
