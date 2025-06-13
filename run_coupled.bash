#!/bin/bash
#
# Script: run_coupled.bash
# Description: Executes a single coupled radiative transfer (HELIOS) and
#              equilibrium chemistry (GGchem) simulation. It iterates
#              between the two codes until convergence is met or
#              the maximum number of coupling iterations is reached.
#
# Usage:
#   bash run_coupled.bash [OPTIONS]
#
# Options:
#   --TOA_P <value>        Top of Atmosphere Pressure (in units of 1e-6 bar). Default: 1e-1
#   --BOA_P <value>        Bottom of Atmosphere Pressure (1e-6 bar). Default: 1e6
#   --TEMP <value>         Internal Temperature (K). Default: 200
#   --ALBEDO <value>       Surface Albedo (dimensionless). Default: 0.1
#   --CplusO <value>       Total Carbon + Oxygen abundance relative to H. Default: 1e-3
#   --CtoO <value>         Carbon-to-Oxygen ratio. Default: 0.59
#   --a_N <value>          Nitrogen abundance. Default: 0.0
#   --i_min <value>        Starting coupling iteration index. Default: 0
#   --OUT_DIR <path>       Output directory for this specific run.
#                          Must be absolute path or relative to CHELIO_PATH.
#   --NAME <string>        Unique name for this simulation run.
#                          Used for output file prefixes.
#
# Environment Variables Required:
#   CHELIO_PATH            Absolute path to the 'chelio' repository root.
#   GGCHEM_PATH            Absolute path to the GGchem installation directory.
#   HELIOS_PATH            Absolute path to the HELIOS installation directory.
#

# --- 1. Script Setup and Path Validation ---

# Exit immediately if a command exits with a non-zero status.
set -e
# Exit if an unset variable is used.
set -u

# Define the absolute path to the chelio repository
# Users are encouraged to set this environment variable:
# export CHELIO_PATH="/absolute/path/to/your/chelio"
#
# If CHELIO_PATH is not already set, attempt to derive it from the current working directory.
if [ -z "${CHELIO_PATH:-}" ]; then
    # If CHELIO_PATH is not set, we assume the user is running the script from the 'chelio' root.
    export CHELIO_PATH="$(pwd)"
fi

# Validate that CHELIO_PATH points to the 'chelio' directory
if [ "$(basename "$CHELIO_PATH")" != "chelio" ]; then
    echo "Error: CHELIO_PATH environment variable is incorrectly set, or"
    echo "       the script is not being run from the 'chelio' directory."
    echo ""
    echo "       Please do ONE of the following:"
    echo "       1. Change your current directory to the 'chelio' repository root (e.g., 'cd /path/to/chelio')"
    echo "          and then run the script."
    echo "       2. Explicitly set the CHELIO_PATH environment variable before running:"
    echo "          'export CHELIO_PATH=/absolute/path/to/your/chelio'"
    echo "          (You might add this line to your ~/.bashrc for persistence)."
    echo ""
    echo "       Current CHELIO_PATH: '${CHELIO_PATH}' (basename: '$(basename "$CHELIO_PATH")')"
    exit 1
fi

# Check if required environment variables are set
if [ -z "${GGCHEM_PATH:-}" ]; then
    echo "Error: GGCHEM_PATH environment variable is not set."
    echo "Please set it to the absolute path of your GGchem installation."
    exit 1
fi
if [ -z "${HELIOS_PATH:-}" ]; then
    echo "Error: HELIOS_PATH environment variable is not set."
    echo "Please set it to the absolute path of your HELIOS installation."
    exit 1
fi

# Check if current directory is 'chelio' or if CHELIO_PATH is valid
if [ "$(basename "$CHELIO_PATH")" != "chelio" ]; then
    echo "Error: Please run this script from the 'chelio' directory, or"
    echo "       ensure CHELIO_PATH is correctly set to its absolute path."
    exit 1
fi

# --- 2. Atmosphere Parameters with Default Values ---

TOA_P=1e-1
BOA_P=1e6
TEMP=200
ALBEDO=0.1
CplusO=1e-3
CtoO=0.59
a_N=0.0
i_min=0 # Starting index for coupling iterations

# Output specific parameters (will be passed from multiple_runs.bash)
OUT_DIR="output" # Output directory relative to CHELIO_PATH
NAME="test" # Name of the simulation
MIXFILE="vertical_mix" # Base name for mixing ratio files

# --- 3. Parse Command-Line Arguments ---

while [ $# -gt 0 ]; do
    case "$1" in
        --TOA_P) TOA_P="$2"; shift 2 ;;
        --BOA_P) BOA_P="$2"; shift 2 ;;
        --TEMP) TEMP="$2"; shift 2 ;;
        --ALBEDO) ALBEDO="$2"; shift 2 ;;
        --CplusO) CplusO="$2"; shift 2 ;;
        --CtoO) CtoO="$2"; shift 2 ;;
        --a_N) a_N="$2"; shift 2 ;;
        --i_min) i_min="$2"; shift 2 ;;
        --OUT_DIR) OUT_DIR="$2"; shift 2 ;;
        --NAME) NAME="$2"; shift 2 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

# --- 4. Prepare Output Directory and Check for Existing Output ---

# Create the specific output directory for this run if it doesn't exist
mkdir -p "${CHELIO_PATH}/${OUT_DIR}"
mkdir -p "${CHELIO_PATH}/${OUT_DIR}/${NAME}"

# Check if the initial output file already exists to prevent overwriting
# This check only considers the very first expected output file.
if [ -f "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_$(($i_min+1)).dat" ]; then
    echo "Warning: Output file ${OUT_DIR}/${NAME}/${MIXFILE}_$(($i_min+1)).dat already exists."
    echo "         Skipping this simulation to prevent overwriting."
    echo "         To force re-run, delete the existing output directory: ${CHELIO_PATH}/${OUT_DIR}/${NAME}"
    exit 0 # Exit with 0 to indicate a "soft" exit (e.g., for batch runs)
fi

echo "--- Starting Chelio Simulation: ${NAME} ---"
echo "Output directory: ${CHELIO_PATH}/${OUT_DIR}"

# --- 5. Initial GGchem Setup and Run ---

echo "Initializing GGchem with initial abundances and P-T profile..."

# Run python script to create initial abundance file for GGchem
python3 "${CHELIO_PATH}/source/calc_abundances.py" \
    --CplusO "$CplusO" --CtoO "$CtoO" --a_N "$a_N"

# Get initial (isothermal) P-T-profile for GGchem
# Starting with higher T (e.g., 500K) can help prevent all species condensing initially
python3 "${CHELIO_PATH}/source/create_pt.py" \
    --Teq 500 --Pmin "$TOA_P" --Pmax "$BOA_P"

# Copy initial input files to the dedicated output directory for archiving
cp "${CHELIO_PATH}/ggchem_inputs/abundances.in" "${CHELIO_PATH}/${OUT_DIR}/${NAME}/."
cp "${CHELIO_PATH}/ggchem_inputs/pt_helios.in" "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_-1.dat"
cp "${CHELIO_PATH}/ggchem_inputs/param.in" "${CHELIO_PATH}/${OUT_DIR}/${NAME}/param_ggchem.in"

# Prepare GGchem's working directory
cp "${CHELIO_PATH}/ggchem_inputs/abundances.in" "${GGCHEM_PATH}/abund_helios.in"
cp "${CHELIO_PATH}/ggchem_inputs/pt_helios.in" "${GGCHEM_PATH}/structures/pt_helios.in"
cp "${CHELIO_PATH}/ggchem_inputs/param.in" "${GGCHEM_PATH}/input/param_helios.in"

ls -la "${GGCHEM_PATH}/input"

# Run GGchem for the first time
echo "Running initial GGchem calculation..."
cd "${GGCHEM_PATH}"
./ggchem input/param_helios.in
cd "${CHELIO_PATH}"

# --- 6. HELIOS-GGchem Coupling Loop ---

# Prepare HELIOS's working directory (param.dat might be modified by HELIOS)
cp "${CHELIO_PATH}/helios_inputs/param.dat" "${HELIOS_PATH}/param.dat"

started_convection=0
coupling_speed_up="no"
i_full=4  # Iteration after which HELIOS runs with full convergence criterion and coupling speed up
i_max=10  # Maximum number of coupling iterations

echo "Starting HELIOS-GGchem coupling iterations (max ${i_max} iterations)..."

for i in $(seq "$i_min" 1 "$i_max"); do
    echo "--- Coupling Iteration: $i ---"

    # Convert GGchem output (Static_Conc.dat) to HELIOS mixfile format
    # This is done if we are starting a new run (i=0) or resuming (i > i_min)
    if (( i > i_min || i == 0 )); then
        echo "Converting GGchem output to HELIOS mixfile..."
        python3 "${CHELIO_PATH}/source/convert_mixfile.py" \
            "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_${i}.dat"
        cp "${GGCHEM_PATH}/Static_Conc.dat" "${CHELIO_PATH}/${OUT_DIR}/${NAME}/Static_Conc_${i}.dat"
    fi

    # Check if the mixfile for the current iteration was successfully created
    if [ ! -f "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_${i}.dat" ]; then
        echo "Error: Vertical mixing ratios file for iteration $i not found: ${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_${i}.dat"
        exit 1
    fi

    # Set HELIOS max iterations based on coupling iteration number
    MAX_ITER=10000 # Default for intermediate iterations
    if (( i == 0 )); then
        MAX_ITER=1000 # Relaxed for first iteration
    elif (( i >= i_full )); then
        MAX_ITER=30000 # Tightest convergence for later iterations
        coupling_speed_up="yes" # Enable HELIOS speed-up (average of last 2 iterations)
    fi

    # Check for previous convection status from HELIOS
    if (( i > 0 )) && [ -f "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_started_convection.dat" ]; then
        started_convection=$(cat "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_started_convection.dat")
        echo "Previous convection status: ${started_convection}"
    fi

    # Run HELIOS
    echo "Running HELIOS for iteration $i..."
    cd "${HELIOS_PATH}"
    python3 -u helios.py \
        -name "${NAME}" \
        -output_directory "${CHELIO_PATH}/${OUT_DIR}/" \
        -toa_pressure "${TOA_P}" \
        -boa_pressure "${BOA_P}" \
        -internal_temperature "${TEMP}" \
        -surface_albedo "${ALBEDO}" \
        -path_to_temperature_file "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_$(($i-1)).dat" \
        -opacity_mixing on-the-fly \
        -path_to_species_file "${CHELIO_PATH}/helios_inputs/species.dat" \
        -file_with_vertical_mixing_ratios "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_${i}.dat" \
        -coupling_mode yes \
        -coupling_iteration_step "$i" \
        -coupling_speed_up "$coupling_speed_up" \
        -started_convection "$started_convection" \
        -write_tp_profile_during_run "$MAX_ITER" \
        -maximum_number_of_iterations "$(($MAX_ITER+1))"
    cd "${CHELIO_PATH}"

    # Check for coupling convergence from HELIOS
    if (( i > 0 )) && [ -f "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_coupling_convergence.dat" ]; then
        STOP=$(cat "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_coupling_convergence.dat")
        echo "--> Coupling converged? ${STOP} (1 = yes, 0 = no)"
        if [[ "${STOP}" -eq 1 ]]; then
            echo "Coupling converged. Stopping iterations."
            break # Exit the loop if converged
        fi
    fi

    # Prepare for next GGchem run: copy new T(P) profile from HELIOS output
    echo "Preparing GGchem for next iteration with new T(P) profile..."
    cp "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_${i}.dat" \
       "${GGCHEM_PATH}/structures/pt_helios.in"

    # Run GGchem with the new T(P) profile
    echo "Running GGchem for iteration $i..."
    cd "${GGCHEM_PATH}"
    ./ggchem input/param_helios.in
    cd "${CHELIO_PATH}"

done

# --- 7. Final Steps ---

# Convert the final GGchem output and save it for analysis
echo "Performing final conversion of GGchem output..."
python3 "${CHELIO_PATH}/source/convert_mixfile.py" \
    "${CHELIO_PATH}/${OUT_DIR}/${NAME}/${MIXFILE}_$(($i+1)).dat"
cp "${GGCHEM_PATH}/Static_Conc.dat" \
   "${CHELIO_PATH}/${OUT_DIR}/${NAME}/Static_Conc_$(($i+1)).dat"

echo "Simulation ${NAME} completed."