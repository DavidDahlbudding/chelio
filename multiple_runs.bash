#!/bin/bash
#
# Script: multiple_runs.bash
# Description: This script orchestrates multiple Chelio simulations by iterating
#              over a defined grid of atmospheric and chemistry parameters.
#              For each parameter combination, it calls 'run_coupled.bash'
#              to execute a single coupled HELIOS-GGchem simulation.
#

# --- Define Parameter Grid ---

# Atmospheric parameters
BOA_Ps=(1e6 1e7 1e8)             # Bottom of Atmosphere Pressure (Pa)
TEMPs=($(seq 50 50 250))         # Internal Temperature (K)

# Chemistry parameters
CplusOs=(1e-3 3.16e-3 1e-2 3.16e-2 1e-1) # C+O abundance relative to H
CtoOs=(0.1 0.59 1.0)             # Carbon-to-Oxygen ratio

# Uncomment and adjust these if you want to include them in the grid later
# a_Ns=(1e-4 1e-2)                 # Nitrogen abundance (example)
# ALBEDOs=(0.0 0.2 0.4 0.8)        # Surface Albedo (example)

# --- Run Parameter Grid ---

# Base output directory relative to 'chelio'
BASE_OUT_DIR="output/EqCond+Remove"

# Create output directory if it does not exist
mkdir -p "${BASE_OUT_DIR}"

echo "Starting parameter grid exploration..."
echo "Output will be saved in: ${BASE_OUT_DIR}/"

for BOA_P in "${BOA_Ps[@]}"; do
    for TEMP in "${TEMPs[@]}"; do
        for CplusO in "${CplusOs[@]}"; do
            for CtoO in "${CtoOs[@]}"; do
                # Construct a unique name for this specific simulation run
                # This name will also be used for its output subdirectory
                SIM_NAME="Earth_P0=${BOA_P}_Tint=${TEMP}_CplusO=${CplusO}_CtoO=${CtoO}" #_aN=${a_N}" #_A=${ALBEDO}"

                # Define the full path for the output directory for this run
                # This directory will be created by 'run_coupled.bash'
                CURRENT_OUT_DIR="${BASE_OUT_DIR}/${SIM_NAME}"

                echo "--- Running simulation for: ${SIM_NAME} ---"

                # Call run_coupled.bash with the specific parameters and output path
                # Note: run_coupled.bash will need to be updated to accept OUT_DIR and NAME
                bash run_coupled.bash \
                    --BOA_P "${BOA_P}" \
                    --TEMP "${TEMP}" \
                    --CplusO "${CplusO}" \
                    --CtoO "${CtoO}" \
                    --OUT_DIR "${CURRENT_OUT_DIR}" \
                    --NAME "${SIM_NAME}"

            done
        done
    done
done

echo "Parameter grid exploration complete!"