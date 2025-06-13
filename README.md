# Chelio: Coupled HELIOS-GGchem Atmospheric Simulation Framework

Chelio is a framework designed to couple the 1D radiative transfer code [HELIOS](https://github.com/exoclime/HELIOS) with the equilibrium chemistry code [GGchem](https://github.com/pw31/GGchem). It enables self-consistent atmospheric simulations by iteratively calculating temperature-pressure profiles and chemical compositions.

---

## Getting Started

### Prerequisites

Before running Chelio, ensure you have the following installed and configured:

* **Python 3**: For Chelio's utility scripts and HELIOS.
* **HELIOS**: The 1D radiative transfer code.
* **GGchem**: The equilibrium chemistry code.

### Environment Setup

You **must** set the following environment variables to the absolute paths of your installations. It's recommended to add these lines to your shell's configuration file (e.g., ~/.bashrc or ~/.zshrc) to make them permanent.

```bash
export CHELIO_PATH="/absolute/path/to/your/chelio"
export GGCHEM_PATH="/absolute/path/to/your/ggchem_installation"
export HELIOS_PATH="/absolute/path/to/your/helios_installation"
```
* **Action**: Replace /absolute/path/to/... with your actual paths.

---

## Usage

All scripts should be run from the root directory of the `chelio` repository (i.e., where README.md is located).

### Running a Single Simulation

To run a single coupled HELIOS-GGchem simulation with specific parameters, use `run_coupled.bash`.

```bash
bash run_coupled.bash \
    --TOA_P 1e-2 \
    --BOA_P 1e7 \
    --TEMP 300 \
    --ALBEDO 0.15 \
    --CplusO 1e-3 \
    --CtoO 0.59 \
    --a_N 1e-4
    # OUT_DIR defaults to "output", NAME defaults to "test"
    # To specify: --OUT_DIR "output/my_single_run" --NAME "MySimulation_P1e7_T300"
```

**Key Parameters (with defaults if not specified):**

* `--TOA_P`: Top of Atmosphere Pressure (in units of 1e-6 bar). Default: 1e-1
* `--BOA_P`: Bottom of Atmosphere Pressure (1e-6 bar). Default: 1e6
* `--TEMP`: Internal Temperature (K). Default: 200
* `--ALBEDO`: Surface Albedo (dimensionless). Default: 0.1
* `--CplusO`: Total Carbon + Oxygen abundance relative to H. Default: 1e-3
* `--CtoO`: Carbon-to-Oxygen ratio. Default: 0.59
* `--a_N`: Nitrogen abundance. Default: 0.0
* `--i_min`: Starting coupling iteration index (useful for resuming runs). Default: 0
* `--OUT_DIR`: Path for the specific run's output directory (relative to `CHELIO_PATH`). This directory will be created if it doesn't exist. Default: "output"
* `--NAME`: A unique name for this simulation, used for output file prefixes. Default: "test"

### Running a Parameter Grid Exploration

To run multiple simulations across a defined parameter space, use `multiple_runs.bash`. This script iterates through arrays of atmospheric and chemical parameters, calling `run_coupled.bash` for each combination.

To modify the parameter ranges, edit the `BOA_Ps`, `TEMPs`, `CplusOs`, and `CtoOs` arrays directly within the `multiple_runs.bash` script.

```bash
bash multiple_runs.bash
```

---

## Project Structure

```
chelio/
├─ README.md               # This file
├─ analyze/                # Jupyter notebooks and generated images for post-processing and analysis
│  ├─ Jupyter Notebooks ...
│  ├─ images/
│  │  ├─ ...
├─ ggchem_inputs/          # Template input files for GGchem
│  ├─ abundances.in         # Initial elemental abundances for GGchem
│  ├─ param.in              # GGchem's main parameter file
│  ├─ param_test.in
│  └─ pt_helios.in          # Initial P-T profile for GGchem
├─ helios_inputs/          # Template input files for HELIOS
│  ├─ mixfile.dat           # Input for HELIOS species mixing ratios
│  ├─ param.dat             # HELIOS's main parameter file (pre-set to Earth-sized moon around Jupiter-like FFP)
│  ├─ param_io.dat          # pre-set parameter file for an Io-sized moon
│  ├─ param_test.dat
│  ├─ species.dat           # List of species for HELIOS
│  └─ species_test.dat
├─ multiple_runs.bash      # Script to run simulations across a parameter grid
├─ run_coupled.bash        # Core script to run a single coupled HELIOS-GGchem simulation
├─ output/                 # Directory where all simulation results are saved
│  ├─ ... (further output or specific run directories, e.g., 'test', ...)
└─ source/                 # Python utility scripts
    ├─ calc_abundances.py     # Calculates initial abundances for GGchem
    ├─ calc_abundances_benchmark.py
    ├─ calc_escape.py         # Calculate stability w.r.t. Jeans escape (assumes const. g and, above top layer, const. T)
    ├─ convert_mixfile.py     # Converts GGchem output to HELIOS mixfile format
    ├─ convert_tp.py          # Converts T(P) profiles from HELIOS to GGchem-readable format
    ├─ create_pt.py           # Creates initial P-T profiles
    └─ mark_bad_last_iters.py # Marks problematic iterations in output
```

---

## How the Coupling Works

The `run_coupled.bash` script orchestrates the iterative coupling between HELIOS and GGchem:

1.  **Initialization**:
    * `source/calc_abundances.py` generates an initial elemental abundance file for GGchem based on input C+O and C:O ratios.
    * `source/create_pt.py` generates an initial (isothermal) Pressure-Temperature (P-T) profile for GGchem.
    * GGchem is run with these initial inputs to calculate an equilibrium chemical composition.

2.  **Coupling Loop**: The script enters a loop, typically running for 10-20 iterations or until convergence:
    * `source/convert_mixfile.py` converts the chemical abundances calculated by GGchem into a format usable by HELIOS (the `vertical_mix.dat` file).
    * HELIOS is executed with the current P-T profile (from the previous iteration or initial) and the new chemical mixing ratios. HELIOS calculates a new P-T profile.
    * The newly calculated P-T profile from HELIOS is then fed back into GGchem.
    * GGchem is run again to compute new chemical abundances based on the updated P-T profile.
    * This cycle continues, with HELIOS and GGchem iteratively updating the atmosphere's P-T profile and chemical composition, respectively, until both converge to a self-consistent state.

3.  **Convergence**: The loop typically breaks when a convergence criterion from HELIOS is met, or after a maximum number of iterations.

---

## Citation

Accompanying paper:

[Your Paper Title]
[Your Authors]
[Journal, Year, Volume, Pages/DOI]