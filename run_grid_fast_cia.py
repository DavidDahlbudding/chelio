#!/usr/bin/env python3
"""
Run a grid of Chelio simulations testing different CIA opacity sources.

For each CIA pair (e.g., N2-CH4), there can be multiple source files (e.g., N2-CH4_2011, N2-CH4_2024).
This script loops over all CIA sources, copies them to the r50_kdistr folder with the proper naming,
runs simulations over different surface pressures and internal temperatures, and restores the default
CIA source after each complete iteration (if needed).
"""
import argparse
import itertools
import subprocess
import os
import shutil
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from chelio_sim.mixfile_utils import mol_dict # for triple and critical point data

# --- Configuration ---
# Path to HELIOS opacity folders (relative to this script's directory or absolute)
HELIOS_PATH = os.environ.get("HELIOS_PATH", os.path.expanduser("~/PhD/HELIOS"))
HITRAN_CIA_DIR = os.path.join(HELIOS_PATH, "input", "opacity", "hitran_cia")
R50_KDISTR_DIR = os.path.join(HELIOS_PATH, "input", "opacity", "r50_kdistr")

# Default CIA sources (from create_all_CIAopacs.bash)
# These are restored after testing non-default sources
DEFAULT_CIA_SOURCES = {
    "CH4-CH4": "CH4-CH4_2011",
    "CO2-CH4": "CO2-CH4_2024_main",
    "CO2-CO2": "CO2-CO2_2024",
    "CO2-H2": "CO2-H2_2024",
    "H2-CH4": "H2-CH4_eq_2011",
    "N2-CH4": "N2-CH4_2024_Tconst=50K",
    "N2-H2": "N2-H2_2024",
    "N2-H2O": "N2-H2O_2018",
    "N2-N2": "N2-N2_2021_Tconst=50K",
    "H2-H2": "H2-H2_2018+02",
}


def get_cia_pair_from_filename(filename):
    """
    Extract the CIA pair from a filename.
    E.g., 'N2-CH4_2024.h5' -> 'N2-CH4'
          'CO2-CH4_2024_main.h5' -> 'CO2-CH4'
    """
    # Remove .h5 extension
    base = filename.replace(".h5", "")
    # Split by underscore and take the first part (the pair)
    parts = base.split("_")
    return parts[0]


def get_cia_target_name(pair):
    """
    Convert CIA pair to target filename in r50_kdistr.
    E.g., 'N2-CH4' -> 'CIA_N2CH4_opac_ip_kdistr.h5'
    """
    pair_no_dash = pair.replace("-", "")
    return f"CIA_{pair_no_dash}_opac_ip_kdistr.h5"


def get_source_name_from_filename(filename):
    """
    Extract the source name (without .h5) from filename.
    E.g., 'N2-CH4_2024.h5' -> 'N2-CH4_2024'
    """
    return filename.replace(".h5", "")


def get_mixing_ratios_for_cia_pair(pair):
    """
    Generate constant mixing ratios for a CIA pair.
    
    For a pair like 'N2-CH4', returns 50% N2 and 50% CH4.
    For self-CIA like 'N2-N2', returns 100% N2.
    
    Always includes N2, CH4, CO2, H2, H2O (with zeros for non-participating species).
    
    Returns a string suitable for --constant_mixing_ratios argument.
    """
    # Base species with zero mixing ratios
    species = {"N2": 0.0, "CH4": 0.0, "CO2": 0.0, "H2": 0.0, "H2O": 1e-30}
    
    # Parse the pair (e.g., "N2-CH4" or "CO2-CO2")
    parts = pair.split("-")
    mol1 = parts[0]
    mol2 = parts[1]
    
    if mol1 == mol2:
        # Self-CIA: 100% of that molecule
        if mol1 in species:
            species[mol1] = 1.0
        else:
            # For species not in our base list, add it
            species[mol1] = 1.0
    else:
        # Cross-CIA: 50% each
        if mol1 in species:
            species[mol1] = 0.5
        else:
            species[mol1] = 0.5
        if mol2 in species:
            species[mol2] = 0.5
        else:
            species[mol2] = 0.5
    
    # Convert to command-line argument string
    return ",".join([f"{k}={v}" for k, v in species.items()])


def copy_cia_to_r50(source_filename):
    """
    Copy a CIA opacity file from hitran_cia to r50_kdistr with proper naming.
    Returns the pair name for later restoration.
    """
    pair = get_cia_pair_from_filename(source_filename)
    source_path = os.path.join(HITRAN_CIA_DIR, source_filename)
    target_name = get_cia_target_name(pair)
    target_path = os.path.join(R50_KDISTR_DIR, target_name)
    
    print(f"Copying {source_filename} -> {target_name}")
    shutil.copy2(source_path, target_path)
    
    return pair


def restore_default_cia(pair):
    """
    Restore the default CIA source for a given pair.
    """
    if pair not in DEFAULT_CIA_SOURCES:
        print(f"Warning: No default CIA source defined for {pair}")
        return
    
    default_source = DEFAULT_CIA_SOURCES[pair]
    source_filename = f"{default_source}.h5"
    source_path = os.path.join(HITRAN_CIA_DIR, source_filename)
    target_name = get_cia_target_name(pair)
    target_path = os.path.join(R50_KDISTR_DIR, target_name)
    
    if os.path.exists(source_path):
        print(f"Restoring default: {source_filename} -> {target_name}")
        shutil.copy2(source_path, target_path)
    else:
        print(f"Warning: Default source {source_filename} not found!")


def run_single_simulation(params):
    """
    Takes a tuple of simulation parameters, constructs the command,
    and executes a single run of run_fast.py.
    """
    temp, psurf, cia_source, cia_pair, base_out_dir = params

    psurf_bar = int(psurf / 1e6)

    # Construct a unique name for the run including the CIA source
    sim_name = f"CIA_{cia_source}_Tint={temp}K_Psurf={psurf_bar}bar"

    # Get mixing ratios based on CIA pair (50/50 for cross-CIA, 100% for self-CIA)
    mixing_ratios_str = get_mixing_ratios_for_cia_pair(cia_pair)

    # The main orchestrator script to call
    #main_script = "./run_fast.py"
    main_script = "./run_coupled.py"

    # Construct the full command as a list of strings
    # Using constant chemistry mode with mixing ratios based on the CIA pair
    command = [
        "python3", main_script,
        "--name", sim_name,
        "--out_dir", base_out_dir,
        "--internal_temp", str(temp),
        "--surface_pressure", str(psurf),
        "--chemistry_mode", "constant",
        "--constant_mixing_ratios", mixing_ratios_str,
        "--outgas_or_manual", "manual",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
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


def run_simulations_for_cia_source(cia_source_filename, cia_pair, temps, psurfs, base_out_dir, n_jobs):
    """
    Run all simulations for a single CIA source file.
    """
    cia_source = get_source_name_from_filename(cia_source_filename)
    
    # Create parameter grid for this CIA source
    param_grid = list(itertools.product(temps, psurfs))
    tasks = [(temp, psurf, cia_source, cia_pair, base_out_dir) for temp, psurf in param_grid]
    total_sims = len(tasks)
    
    # Show the mixing ratios being used
    mixing_ratios_str = get_mixing_ratios_for_cia_pair(cia_pair)
    print(f"\n--- Running {total_sims} simulations for CIA source: {cia_source} ---")
    print(f"    Mixing ratios: {mixing_ratios_str}")
    
    completed_count = 0
    failed_runs = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=total_sims, desc=f"CIA: {cia_source}"):
            sim_name, success, message = future.result()
            completed_count += 1
            if success:
                print(f"  ({completed_count}/{total_sims}) COMPLETED: {sim_name}")
            else:
                print(f"  ({completed_count}/{total_sims}) FAILED: {sim_name}")
                print(message)
                failed_runs.append(sim_name)
    
    return failed_runs


def discover_cia_sources():
    """
    Discover all CIA source files in the hitran_cia directory.
    Returns a dict: {pair: [list of source filenames]}
    """
    cia_files = glob.glob(os.path.join(HITRAN_CIA_DIR, "*.h5"))
    cia_files = [os.path.basename(f) for f in cia_files]
    
    # Group by pair
    sources_by_pair = {}
    for filename in sorted(cia_files):
        pair = get_cia_pair_from_filename(filename)
        if pair not in sources_by_pair:
            sources_by_pair[pair] = []
        sources_by_pair[pair].append(filename)
    
    return sources_by_pair


# --- Main script execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid of Chelio simulations testing different CIA opacity sources."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of simulations to run in parallel."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Specific CIA pairs to test (e.g., N2-CH4 CO2-CO2). Default: all pairs."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Specific CIA source filename to test (e.g., N2-CH4_2024.h5). Overrides --pairs if specified."
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available CIA sources and exit."
    )
    parser.add_argument(
        "--restore-defaults",
        action="store_true",
        help="Restore default CIA sources before starting (useful if previous runs didn't restore)."
    )
    args = parser.parse_args()

    # Restore defaults before starting (in case of previous runs that didn't restore)
    print("Restoring default CIA sources before starting...")
    for pair in DEFAULT_CIA_SOURCES.keys():
        restore_default_cia(pair)
    
    if args.restore_defaults:
        print("Default CIA sources have been restored. Exiting as per --restore-defaults flag.")
        exit(0)

    # Discover available CIA sources
    sources_by_pair = discover_cia_sources()
    
    if args.list_sources:
        print("Available CIA sources by pair:")
        for pair, sources in sorted(sources_by_pair.items()):
            default_marker = f" [default: {DEFAULT_CIA_SOURCES.get(pair, 'N/A')}]"
            print(f"\n{pair}{default_marker}:")
            for src in sources:
                is_default = get_source_name_from_filename(src) == DEFAULT_CIA_SOURCES.get(pair)
                marker = " *" if is_default else ""
                print(f"  - {src}{marker}")
        exit(0)

    # --- Define Parameter Grid ---
    temps = [150] # K
    psurfs = [1e8]  # dyn/cm^2

    base_out_dir = "output/CIA_comparison_HELIOS"

    # Filter pairs if specified
    if args.source:
        # If a specific source is given, find its pair and only test that pair
        source_filename = args.source
        pair = get_cia_pair_from_filename(source_filename)
        if pair not in sources_by_pair or source_filename not in sources_by_pair[pair]:
            print(f"Error: Specified source '{source_filename}' not found in hitran_cia directory.")
            exit(1)
        pairs_to_test = [pair]
        sources_by_pair[pair] = [source_filename]  # Only test the specified source for that pair
    elif args.pairs:
        pairs_to_test = args.pairs
    else:
        pairs_to_test = list(sources_by_pair.keys())

    # Validate requested pairs
    for pair in pairs_to_test:
        if pair not in sources_by_pair:
            print(f"Error: CIA pair '{pair}' not found. Available pairs: {list(sources_by_pair.keys())}")
            exit(1)

    # --- Summary ---
    total_sources = sum(len(sources_by_pair[p]) for p in pairs_to_test)
    total_sims_per_source = len(temps) * len(psurfs)
    total_sims = total_sources * total_sims_per_source
    
    print(f"=== CIA Opacity Grid Exploration ===")
    print(f"CIA pairs to test: {pairs_to_test}")
    print(f"Total CIA sources: {total_sources}")
    print(f"Simulations per source: {total_sims_per_source}")
    print(f"Total simulations: {total_sims}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Output directory: {base_out_dir}")
    print()

    # --- Run simulations for each CIA source ---
    all_failed_runs = []
    
    for pair in pairs_to_test:
        sources = sources_by_pair[pair]
        print(f"\n{'='*60}")
        print(f"Processing CIA pair: {pair} ({len(sources)} source(s))")
        print(f"{'='*60}")

        psurfs_pair = psurfs.copy()
        if 'critical' in psurfs_pair:
            p_crit_min = [mol_dict[mol]['critical'][1] for mol in pair.split('-')]

            if p_crit_min[0] < p_crit_min[1]:
                p_crit_min = p_crit_min[0]
            else:
                p_crit_min = p_crit_min[1]
            
            i_crit = psurfs_pair.index('critical')
            psurfs_pair[i_crit] = float(p_crit_min)
        
        for source_filename in sources:
            source_name = get_source_name_from_filename(source_filename)
            
            # Copy the CIA source to r50_kdistr
            copy_cia_to_r50(source_filename)
            
            # Run all simulations for this source (pass the pair for mixing ratio calculation)
            failed = run_simulations_for_cia_source(
                source_filename, pair, temps, psurfs_pair, base_out_dir, args.jobs
            )
            all_failed_runs.extend(failed)
            
            # Check if we need to restore the default
            default_source = DEFAULT_CIA_SOURCES.get(pair)
            if default_source and source_name != default_source:
                print(f"Restoring default CIA source for {pair}...")
                restore_default_cia(pair)
            else:
                print(f"Source {source_name} is the default for {pair}, no restoration needed.")

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print("=== CIA Grid Exploration Complete! ===")
    print(f"{'='*60}")
    print(f"Total simulations run: {total_sims}")
    print(f"Successful: {total_sims - len(all_failed_runs)}")
    print(f"Failed: {len(all_failed_runs)}")
    
    if all_failed_runs:
        print("\nFailed simulation names:")
        for name in all_failed_runs:
            print(f"  - {name}")

