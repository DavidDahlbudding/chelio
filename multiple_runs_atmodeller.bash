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
TEMPs=($(seq 50 50 200))         # Internal Temperature (K)

# Chemistry parameters
TEMP_MELTS=(1000 2000) # Melting temperature (K)
#MELT_FRACS=(0.1 1.0) # Melting fraction
H_OCEANS=(0.1 1.0 10.0) # H ocean abundance
CtoHs=(0.1 1.0 10.0) # C/H mass ratio
NtoCs=(0.01 0.1 1.0) # N/C mass ratio
fO2s=(-5 -2.5 0 2.5 5) # Oxygen fugacity fO2 [delta IW]

# --- Run Parameter Grid ---

# Base output directory relative to 'chelio'
BASE_OUT_DIR="output/Atmodeller"

# Create output directory if it does not exist
mkdir -p "${BASE_OUT_DIR}"

echo "Starting parameter grid exploration..."
echo "Output will be saved in: ${BASE_OUT_DIR}/"


for TEMP in "${TEMPs[@]}"; do
    for TEMP_MELT in "${TEMP_MELTS[@]}"; do
        for H_OCEAN in "${H_OCEANS[@]}"; do
            for CtoH in "${CtoHs[@]}"; do
                for NtoC in "${NtoCs[@]}"; do
                    for fO2 in "${fO2s[@]}"; do
                        # Construct a unique name for this specific simulation run
                        # This name will also be used for its output subdirectory
                        SIM_NAME="Earth_Tint=${TEMP}_Tmelt=${TEMP_MELT}_Hocean=${H_OCEAN}_CtoH=${CtoH}_NtoC=${NtoC}_fO2=${fO2}" #_meltFrac=${MELT_FRAC}" #_A=${ALBEDO}"

                        # Define the full path for the output directory for this run
                        # This directory will be created by 'run_coupled.bash'
                        CURRENT_OUT_DIR="${BASE_OUT_DIR}" #/${SIM_NAME}"

                        echo "--- Running simulation for: ${SIM_NAME} ---"

                        # Call run_coupled.bash with the specific parameters and output path
                        # Note: run_coupled.bash will need to be updated to accept OUT_DIR and NAME
                        bash run_coupled_atmodeller.bash \
                            --TEMP "${TEMP}" \
                            --TEMP_MELT "${TEMP_MELT}" \
                            --H_OCEAN "${H_OCEAN}" \
                            --CtoH "${CtoH}" \
                            --NtoC "${NtoC}" \
                            --fO2 "${fO2}" \
                            --OUT_DIR "${CURRENT_OUT_DIR}" \
                            --NAME "${SIM_NAME}" \
                            --with_ggchem "True"
                        
                    done
                done
            done
        done
    done
done

echo "Parameter grid exploration complete!"