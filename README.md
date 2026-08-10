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
export GGCHEM_PATH="/absolute/path/to/your/ggchem_installation"
export HELIOS_PATH="/absolute/path/to/your/helios_installation"
```
* **Action**: Replace /absolute/path/to/... with your actual paths.

Alternatively, you can edit the `config.yaml` file directly and replace `${GGCHEM_PATH}` and `${HELIOS_PATH}` with the absolute paths.

---

## Usage

All simulations are now run through the main `run_coupled.py` script. All scripts should be run from the root directory of the `chelio` repository.

### Running a Single Simulation

To run a single coupled HELIOS-GGchem simulation, use `run_coupled.py` with a unique name for the run.

```bash
python3 run_coupled.py --name "TestRun"
```

This will run a simulation named "TestRun" using the default parameters specified in `config.yaml`.

### Overriding Parameters

You can override any simulation parameter from the config file via command-line flags. For example, to run a simulation with a different internal temperature:

```bash
python3 run_coupled.py --name "HotPlanetRun" --internal_temp 300
```
This is equivalent to the old `run_coupled.bash` script but offers more flexibility through the central `config.yaml` file.

### Running a Parameter Grid Exploration

To run multiple simulations across a defined parameter space, you can use a simple bash loop that calls `run_coupled.py`. For example, to iterate over different C/H ratios:

```bash
for CTOH_RATIO in 0.5 1.0 1.5; do
    RUN_NAME="MyRun_CtoH_${CTOH_RATIO}"
    python3 run_coupled.py --name "${RUN_NAME}" --ctoh_ratio "${CTOH_RATIO}"
done
```
This replaces the functionality of the old `multiple_runs.bash` script.

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
├─ run_coupled.py          # Core script to run a single coupled HELIOS-GGchem simulation
├─ run_fast.py             # Use approximate fast RT with Rosseland mean opacities
├─ run_grid_fast_cia.py    # Loop over P-T-grid for one or more CIA pair (see 2nd paper for details)
├─ run_grid_EarlyMars_cia.py    # Benchmark Code against Turbet+ (2020)
├─ config.yaml             # Central configuration file for all simulations
├─ output/                 # Directory where all simulation results are saved
│  ├─ ... (further output or specific run directories, e.g., 'test', ...)
└─ chelio_sim/             # Python package for simulation logic and utilities
    ├─ __init__.py
    ├─ external_runners.py # Wrappers for HELIOS and GGchem
    ├─ abundances.py       # Calculates initial abundances
    ├─ init_pt.py          # Creates initial P-T profiles
    ├─ mixfile_utils.py    # Converts GGchem output to HELIOS mixfile format
    ├─ rt_utils.py         # Approximate fast RT with on-the-fly Rosseland mean calculation*
    ├─ ... (other utilities)
```

\* parts taken from [Roccetti+ (2023)](https://doi.org/10.1017/S1473550423000046), [(see code)](https://github.com/giulia-roccetti/Master_Thesis/blob/main/tsurf.py)

---

## Citation

Accompanying paper:

[Habitability of Tidally Heated H$_2$-Dominated Exomoons around Free-Floating Planets\
Dahlbüdding et al. (2026)](https://doi.org/10.1093/mnras/stag243)

Also used in:

Small Collisions, High Impact: The Sensitivity of Atmospheric Temperatures to Collision-Induced Absorption
Dahlbüdding et al. (subm.)
