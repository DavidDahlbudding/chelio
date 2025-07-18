# Chelio: Coupled HELIOS-GGchem Atmospheric Simulation Framework

Chelio is a framework designed to couple the 1D radiative transfer code [HELIOS](https://github.com/exoclime/HELIOS) with the equilibrium chemistry code [GGchem](https://github.com/pw31/GGchem). It enables self-consistent atmospheric simulations by iteratively calculating temperature-pressure profiles and chemical compositions.

This framework accompanies the paper "Habitability of Tidally Heated H$_2$-Dominated Exomoons around Free-Floating Planets" by Dahlbüdding et al. (subm.). The data produced by Chelio and presented in the paper are available on [Zenodo](https://doi.org/10.5281/zenodo.15738536).

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
    # To specify, e.g.: --OUT_DIR "output/eqChem" --NAME "Earth_P0=1e7_Tint=200_CtoO=0.1"
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
* `--OUT_DIR`: Path for the general output directory (relative to `CHELIO_PATH`). This directory will be created if it doesn't exist. Default: "output"
* `--NAME`: A unique name for this simulation, used for output directory of a specific run and file prefixes. Default: "test"

### Running a Parameter Grid Exploration

To run multiple simulations across a defined parameter space, use `multiple_runs.bash`. This script iterates through arrays of atmospheric and chemical parameters, calling `run_coupled.bash` for each combination.

To modify the parameter ranges, edit the `BOA_Ps`, `TEMPs`, `CplusOs`, and `CtoOs` arrays directly within the `multiple_runs.bash` script.

```bash
bash multiple_runs.bash
```

---

## Analyzing Simulation Data

The `analyze/` directory contains Jupyter notebooks for post-processing and visualizing simulation results. The analysis workflow is powered by the `analyze_modules` package, which provides a streamlined interface for loading and plotting data.

The notebooks provide templates for common analysis tasks:

1.  **IndividualRun:** Analyze the temperature and chemical profiles of an individual run.
2.  **CompareTsurf+TimeinHZ:** Compare 1D surface temperature vs. a varying parameter and plot histograms of time spent in the habitable zone (valid for Earth-sized moons).
3.  **TsurfMatrix:** Plot 2D matrices of surface temperature or other parameters as a function of chemical composition (C+O, C/O).
4.  **CompareOther:** Create 1D comparison plots for various output parameters, such as surface mixing ratios vs. an input parameter.
5.  **EscapeStatistics:** Generate histograms of the Jeans escape parameter and atmospheric escape timescales.

---

## Project Structure

```
chelio/
├─ README.md               # This file
├─ analyze/                # Analysis tools, notebooks, and figures
│  ├─ 1_IndividualRun.ipynb
│  ├─ ... (other notebooks)
│  ├─ analyze_modules/      # Core package for data analysis
│  │  ├─ __init__.py
│  │  ├─ data_loader.py    # High-level functions for loading simulation data
│  │  └─ plot_utils.py     # Reusable, high-level plotting functions
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

## Citation

Accompanying paper:

Habitability of Tidally Heated H$_2$-Dominated Exomoons around Free-Floating Planets

Dahlbüdding et al. (subm.)