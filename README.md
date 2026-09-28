# Chelio: Coupled HELIOS-GGchem Atmospheric Simulation Framework

Chelio is a framework designed to couple the 1D radiative transfer code [HELIOS](https://github.com/exoclime/HELIOS) with the equilibrium chemistry code [GGchem](https://github.com/pw31/GGchem). It enables self-consistent atmospheric simulations by iteratively calculating temperature-pressure profiles and chemical compositions.

This framework accompanies the paper "Habitability of Tidally Heated H$_2$-Dominated Exomoons around Free-Floating Planets" by [Dahlbüdding et al. (2026)](https://doi.org/10.1093/mnras/stag243). The data produced by Chelio and presented in the paper are available on [Zenodo](https://doi.org/10.5281/zenodo.15738536).

---

## How It Works

Chelio is an orchestrator: it writes input files, calls HELIOS and GGchem as external programs, and converts between their formats.

1. **Initial composition**: elemental abundances are set manually (`a_h`, `a_c`, `a_o`, `a_n`) or derived from magma-ocean outgassing with [atmodeller](https://github.com/ExPlanetology/atmodeller). GGchem then computes the equilibrium composition on an initial T-P profile.
2. **Coupling loop**: the composition is converted to a HELIOS mixing-ratio file (with condensation limits and an adiabatic-gradient table). HELIOS computes a new radiative-convective T-P profile, and GGchem re-equilibrates the chemistry on it.
3. The loop repeats until HELIOS reports convergence or `i_max` is reached.

Alternatively, `--chemistry_mode constant` skips GGchem and uses fixed volume mixing ratios, capped by the saturation vapour pressure. This mode is used for the CIA studies.

---

## Getting Started

### Prerequisites

Before running Chelio, ensure you have the following installed and configured:

* **Python 3** with the packages in `requirements.txt` (`pip install -r requirements.txt`).
* **HELIOS**: the 1D radiative transfer code. Chelio requires the modified version at [DavidDahlbudding/HELIOS](https://github.com/DavidDahlbudding/HELIOS). It adds the coupling flags Chelio passes to HELIOS: `coupling_speed_up`, `started_convection` and the `kappa_file_path` table input.
* **GGchem**: the equilibrium chemistry code (compiled `./ggchem` executable).

### Environment Setup

You **must** set the following environment variables to absolute paths. It's recommended to add these lines to your shell's configuration file (e.g., ~/.bashrc or ~/.zshrc) to make them permanent.

```bash
export GGCHEM_PATH="/absolute/path/to/your/ggchem_installation"
export HELIOS_PATH="/absolute/path/to/your/helios_installation"
export CHELIO_PATH="/absolute/path/to/chelio"   # this repository
```

For the two code paths, you can alternatively edit `config.yaml` and replace `${GGCHEM_PATH}` and `${HELIOS_PATH}` with absolute paths. `chelio_sim` itself also reads all three variables from the environment, so they must be set either way.

---

## Usage

All scripts must be run from the root directory of the `chelio` repository. Single runs go through `run_coupled.py`, and the `run_grid_*.py` scripts call it repeatedly (see below).

Note that pressures in `config.yaml` and on the command line are in **dyn/cm²** (1e6 = 1 bar). The T-P files written to `output/` are in bar.

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
Every numeric key in `simulation_params` (e.g. `--surface_pressure`, `--a_c`, `--surface_albedo`) can be overridden this way. Iteration settings (`i_min`, `i_max`, `i_full_convergence`, HELIOS iteration limits) live in the `coupling` section of `config.yaml`.

### Constant Chemistry Mode

```bash
python3 run_coupled.py --name "N2CH4" --chemistry_mode constant \
    --constant_mixing_ratios "N2=0.5,CH4=0.5,CO2=0,H2=0,H2O=0" \
    --surface_pressure 1e7 --relative_humidity 0.8
```

### Outputs and Restarts

Each run writes to `output/<name>/`. The main files are:

* `<name>_tp_coupling_<i>.dat`: T-P profile after iteration `i` (`-1` is the initial guess).
* `vertical_mix_<i>.dat` and `Static_Conc_<i>.dat`: HELIOS mixing ratios and GGchem output.
* `delad_chelio.dat`: adiabatic-gradient table read by HELIOS. There is a single file, overwritten every iteration.
* `run.log`, plus the standard HELIOS output files.

`run_coupled.py` refuses to overwrite a finished run. To resume an interrupted run, set `coupling.i_min` in `config.yaml` to the next iteration; the run then restarts from `<name>_tp_coupling_<i_min-1>.dat`. `--init_pt_file` starts a new run from any existing T-P profile.

### Running a Parameter Grid Exploration

The grid scripts run many `run_coupled.py` (or `run_fast.py`) calls, one after another. Their parameter lists and output directories are hard-coded at the top of each script's `__main__` block, so edit them there.

| Script | Purpose |
|---|---|
| `run_grid_manual.py` | Grid over T_int × P_surf × manual elemental abundances, using `run_coupled.py`. |
| `run_grid_fast.py` | The same grid using the fast approximate RT (`run_fast.py`). |
| `run_grid_atmodeller.py` | Grid for outgassed atmospheres. Currently only `--internal_temp` is passed; the outgassing parameters are taken from `config.yaml`. |
| `run_grid_fast_cia.py` | Loops over one or more CIA pairs and every available CIA opacity source (see 2nd paper). Each source is swapped into HELIOS' `r50_kdistr` in turn, constant-chemistry runs are made, and the defaults are restored afterwards. Options: `--pairs`, `--source`, `--list-sources`, `--restore-defaults`. |
| `run_grid_EarlyMars_cia.py` | Benchmark against Turbet+ (2020): 2 bar CO₂ with 80 % RH H₂O and trace H₂/CH₄, for different CO₂–H₂ / CO₂–CH₄ CIA sources. It needs `helios_inputs/param_EarlyMars.dat`. |

**Caution:** runs share working files, both in `ggchem_inputs/` and `helios_inputs/` and inside the HELIOS/GGchem installations. Simulations therefore must not run in parallel, and the grid scripts deliberately run them sequentially. This includes starting several `run_coupled.py` processes by hand.

For a quick ad-hoc sweep, a bash loop works as well:

```bash
for T in 100 200 300; do
    python3 run_coupled.py --name "Grid_T${T}" --internal_temp ${T}
done
```

### Other Tools

* `run_fast.py`: same interface as `run_coupled.py`, but HELIOS is replaced by an approximate, much faster RT scheme that uses Rosseland mean opacities (`chelio_sim/rt_utils.py`).
* `get_last_ggchem.py --folder output/<grid> --ggchem_path $GGCHEM_PATH`: for every run in a folder, computes the GGchem composition for the last T-P profile if it is missing (e.g. for aborted runs).

---

## Analyzing Simulation Data

The `analyze/` directory contains Jupyter notebooks for post-processing and visualizing simulation results. The analysis workflow is powered by the `analyze_modules` package, which provides a streamlined interface for loading and plotting data.

The notebooks provide templates for common analysis tasks:

1.  **IndividualRun:** Analyze the temperature and chemical profiles of an individual run.
2.  **CompareTsurf+TimeinHZ:** Compare 1D surface temperature vs. a varying parameter and plot histograms of time spent in the habitable zone (valid for Earth-sized moons).
3.  **TsurfMatrix:** Plot 2D matrices of surface temperature or other parameters as a function of chemical composition (C+O, C/O).
4.  **CompareOther:** Create 1D comparison plots for various output parameters, such as surface mixing ratios vs. an input parameter.
5.  **EscapeStatistics:** Generate histograms of the Jeans escape parameter and atmospheric escape timescales.
6.  **cornerplots:** Corner plots of surface conditions across a parameter grid.

For the CIA study, `cia-opac_relevance.ipynb` and the CLI script `opac_relevance.py` (e.g. `python3 analyze/opac_relevance.py --pair CO2-CH4 --Tmax 600`) map where in T-P space a CIA pair dominates the opacity.

---

## Project Structure

```
chelio/
├─ README.md
├─ CLAUDE.md                   # Developer notes (architecture, pitfalls)
├─ config.yaml                 # Central configuration file for all simulations
├─ requirements.txt
├─ run_coupled.py              # Core script: single coupled HELIOS-GGchem (or constant-chemistry) run
├─ run_fast.py                 # Same, with approximate fast RT (Rosseland mean opacities) instead of HELIOS
├─ run_grid_manual.py          # Grid: T_int × P_surf × manual abundances
├─ run_grid_fast.py            # Same grid using run_fast.py
├─ run_grid_atmodeller.py      # Grid for outgassed atmospheres
├─ run_grid_fast_cia.py        # Loop over P-T grid for one or more CIA pairs (see 2nd paper for details)
├─ run_grid_EarlyMars_cia.py   # Benchmark code against Turbet+ (2020)
├─ get_last_ggchem.py          # Compute missing final GGchem output for finished/aborted runs
├─ analyze/                    # Analysis notebooks and tools
│  ├─ 1_IndividualRun.ipynb, 2_…, 3_…, A_…, B_…, C_cornerplots.ipynb
│  ├─ cia-opac_relevance.ipynb, opac_relevance.py
│  └─ analyze_modules/         # data_loader.py (ChelioRun, sweep/matrix loaders), plot_utils.py
├─ chelio_sim/                 # Python package for simulation logic and utilities
│  ├─ external_runners.py      # Wrappers for HELIOS and GGchem
│  ├─ abundances.py            # Initial abundances (manual or atmodeller outgassing)
│  ├─ init_pt.py               # Initial P-T profiles
│  ├─ mixfile_utils.py         # GGchem → HELIOS mixfile, condensation limits, adiabatic-gradient table
│  ├─ rt_utils.py              # Approximate fast RT with on-the-fly Rosseland mean calculation*
│  └─ ...                      # Standalone helpers (escape, abundance and T-P conversion scripts)
├─ ggchem_inputs/
│  └─ param.in                 # GGchem's main parameter file (abundances.in, pt_helios.in are generated here)
├─ helios_inputs/
│  ├─ param.dat                # HELIOS's main parameter file (pre-set to Earth-sized moon around Jupiter-like FFP)
│  └─ species.dat              # List of species for HELIOS (P_BOA.dat is generated here)
├─ output/                     # All simulation results, one directory per run
└─ .archive/                   # Legacy bash drivers, replaced by the Python scripts
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
