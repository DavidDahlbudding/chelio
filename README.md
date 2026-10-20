# Chelio: Coupled HELIOS-GGchem Atmospheric Simulation Framework

Chelio is a framework designed to couple the 1D radiative transfer code [HELIOS](https://github.com/exoclime/HELIOS) with the equilibrium chemistry code [GGchem](https://github.com/pw31/GGchem). It enables self-consistent atmospheric simulations by iteratively calculating temperature-pressure profiles and chemical compositions.

This framework accompanies the paper "Habitability of Tidally Heated H$_2$-Dominated Exomoons around Free-Floating Planets" by [Dahlbüdding et al. (2026)](https://doi.org/10.1093/mnras/stag243). The data produced by Chelio and presented in the paper are available on [Zenodo](https://doi.org/10.5281/zenodo.15738536).

---

## How It Works

Chelio is an orchestrator: it writes input files, calls HELIOS and GGchem as external programs, and converts between their formats.

1. **Initial composition**: elemental abundances are set manually (`a_h`, `a_c`, `a_o`, `a_n`) or derived from magma-ocean outgassing with [atmodeller](https://github.com/ExPlanetology/atmodeller). GGchem then computes the equilibrium composition on an initial T-P profile.
2. **Coupling loop**: the composition is converted to a HELIOS mixing-ratio file (with condensation limits and an adiabatic-gradient table). HELIOS computes a new radiative-convective T-P profile, and GGchem re-equilibrates the chemistry on it.
3. The loop repeats until HELIOS reports convergence or `i_max` is reached.

Alternatively, `--chemistry constant` skips GGchem and uses fixed volume mixing ratios, capped by the saturation vapour pressure. This mode is used for the CIA studies.

---

## Getting Started

### Prerequisites

Before running Chelio, ensure you have the following installed and configured:

* **Python 3** with the packages in `requirements.txt` (`pip install -r requirements.txt`).
* **HELIOS**: the 1D radiative transfer code. Chelio requires the modified version at [DavidDahlbudding/HELIOS](https://github.com/DavidDahlbudding/HELIOS). It adds the coupling flags Chelio passes to HELIOS: `coupling_speed_up`, `started_convection` and the `kappa_file_path` table input. For clouds, use a version with the fixed cloud asymmetry parameter (branch `fix-cloud-asymmetry-parameter`, see [Clouds](#clouds)).
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

All scripts must be run from the root directory of the `chelio` repository. Single runs go through `run_coupled.py`, and `run_grid.py` calls it repeatedly for a parameter grid (see below).

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

### Run Modes

Three switches in `simulation_params` (each also a CLI flag) select what a run does:

| Setting | Values | Meaning |
|---|---|---|
| `chemistry` | `equilibrium` (default), `constant` | GGchem equilibrium chemistry, or fixed `constant_mixing_ratios` capped by saturation (no GGchem). |
| `outgas_or_manual` | `manual` (default), `outgas` | Source of the elemental abundances for equilibrium chemistry: `a_h`…`a_n` + `surface_pressure`, or atmodeller outgassing (which also sets the surface pressure). |
| `uncoupled_outgassed_run` | `none` (default), `also`, `only` | Extra HELIOS run with the outgassed composition held fixed (no GGchem), written to `output/<name>_outgassed/`; `also` runs it before the coupled run, `only` instead of it. |

Settings that the chosen combination ignores produce a warning at the start of `run.log`, e.g. outgassing parameters with `constant` chemistry, `uncoupled_outgassed_run also` without `outgas`, or `--constant_mixing_ratios` / `--relative_humidity` with `equilibrium` chemistry. `--relative_humidity` only scales the H₂O saturation cap, so it also warns when H₂O is not in the mixture. The run stops with an explanation only if nothing sensible can be run, e.g. `uncoupled_outgassed_run only` without `outgas`, or missing or invalid `constant_mixing_ratios`.

### Constant Chemistry Mode

```bash
python3 run_coupled.py --name "N2CH4" --chemistry constant \
    --constant_mixing_ratios "N2=0.5,CH4=0.5,CO2=0,H2=0,H2O=0" \
    --surface_pressure 1e7 --relative_humidity 0.8
```

### Outputs and Restarts

Each run writes to `output/<name>/`. The main files are:

* `<name>_tp_coupling_<i>.dat`: T-P profile after iteration `i` (`-1` is the initial guess).
* `vertical_mix_<i>.dat` and `Static_Conc_<i>.dat`: HELIOS mixing ratios and GGchem output.
* `delad_chelio.dat`: adiabatic-gradient table read by HELIOS. There is a single file, overwritten every iteration.
* `run.log`, plus the standard HELIOS output files.

`run_coupled.py` refuses to overwrite a finished run. To resume an interrupted run, set `coupling.i_min` in `config.yaml` to the next iteration; the run then restarts from `<name>_tp_coupling_<i_min-1>.dat`. `--init_pt_file` starts a new run from any existing T-P profile, and `--helios_param_file` uses a different HELIOS parameter file than `helios_inputs/param.dat`.

### Active Species

HELIOS only loads the opacities of the *active* species of a run, since reading opacities takes most of its start-up time. `helios_inputs/all_species.dat` is the catalog of available species; the selection is set by `simulation_params.active_species` in `config.yaml` (or `--active_species`, also usable in grid files):

* `auto` (default): for constant chemistry, the species in `constant_mixing_ratios` (with a VMR ≥ 1e-29); for equilibrium chemistry with manual abundances, every catalog species made of elements with abundance > 0 (e.g. no NH₃, HCN, N₂ if `a_n` is 0); for outgassed atmospheres, the species outgassed by atmodeller.
* a list, e.g. `[H2O, CO2, CH4, H2]` (command line: `--active_species H2O,CO2,CH4,H2`).

CIA pairs are never listed: a pair is included exactly when both collision partners are active. H₂O is always active, because HELIOS needs it; with constant chemistry it gets a VMR of 1e-30 if `constant_mixing_ratios` does not list it. The resulting species file is written to `output/<name>/species.dat`.

### Clouds

`run_coupled.py` can add one HELIOS cloud deck in HELIOS' `manual` mode: no cloud below the cloud base `p_bot`, and a cloud mixing ratio (particles per gas molecule) `f = f_bot (p / p_bot)^(1/h - 1)` above it. The particles follow a lognormal size distribution, and their optical properties come from Mie tables. Runs stay cloud-free as long as `cloud_bottom_mixing_ratio` is 0, which is the default. The settings live in `simulation_params` of `config.yaml`, and each one can be overridden on the command line (`--cloud_bottom_pressure 1e5`, …):

| Key | Meaning |
|---|---|
| `cloud_mie_path` | directory with the Mie tables (absolute, or relative to the Chelio root) |
| `cloud_r_mode` | mode of the lognormal dN/dr [µm] (default 2) |
| `cloud_r_std_dev` | geometric standard deviation σ_g, > 1 (default 2) |
| `cloud_bottom_pressure` | cloud base [dyn/cm²], must lie inside the atmosphere |
| `cloud_bottom_mixing_ratio` | f_bot, cloud particles per gas molecule (0 = no cloud) |
| `cloud_to_gas_scale_height_ratio` | h (default 0.5) |

Every HELIOS call gets `-number_of_cloud_decks` (0 without a cloud), so the cloud section of `helios_inputs/param.dat` never matters. A cloudy run logs a `Clouds: …` line and writes `cloud_settings.json` (f_bot, nominal optical depth, Planck-mean cross section, …) to its output directory. `run_fast.py` has no clouds and stops if `cloud_bottom_mixing_ratio` > 0.

**Mie tables.** `chelio_sim/make_mie_files.py` computes them with LX-MIE (`pip install lxmie`) from a refractive-index file in the format of the [LX-MIE compilation](https://github.com/NewStrangeWorlds/LX-MIE/tree/master/compilation). Water ice (Warren 1984) is included in `helios_inputs/clouds/refractive_index/H2O_s.dat`:

```bash
python3 -m chelio_sim.make_mie_files --ri helios_inputs/clouds/refractive_index/H2O_s.dat --out helios_inputs/clouds/H2O_s/
python3 -m chelio_sim.make_mie_files --check --out helios_inputs/clouds/H2O_s/   # validate, plot to analyze/images/
```

The tables (about 12 MB, not tracked by git) cover 0.8× the smallest to 1.25× the largest HELIOS bin edge of `$HELIOS_PATH/input/opacity/r50_kdistr/`. HELIOS gives bins outside the Mie wavelength range zero cloud opacity without a warning. `meta.json` in the output directory records the inputs. `--const_nk N K` instead of `--ri` uses a constant refractive index, for debugging only.

**Optical depth instead of f_bot.** f_bot (around 1e-20) is hard to choose directly, and the same f_bot gives an optical depth proportional to p_bot. In grid files, the derive hook `cloud_from_tau` therefore takes the labels `cloud_tau` (total optical depth, 0 = cloud-free) and `cloud_pbot_bar` (cloud base in bar), and sets f_bot = τ μ m_u g / (p_bot h C̄_ext). Here C̄_ext is the Planck mean at T_int of the size-averaged extinction cross section, μ = 2.12 and g = `planet_params.g`. The same functions are in `chelio_sim/clouds.py`.

`grids/cloud_test.yaml` is the first test: an H₂O-ice deck at 1, 0.1 and 0.01 bar with τ = 0.1, 1 and 10, plus a cloud-free baseline, in the representative case of Dahlbüdding et al. (2026). `python3 analyze/cloud_test.py [--checks]` makes the summary table and figures.

**HELIOS fix.** HELIOS (upstream `master` and `development`) averaged the cloud asymmetry parameter g wrongly over the size distribution: it used the mean scattering cross section in cm² as g, so clouds scattered isotropically (g ≈ 1e-6). The fork's branch `fix-cloud-asymmetry-parameter` uses the scattering-weighted mean g instead; `make_mie_files --check` tests the HELIOS in `$HELIOS_PATH` for it.

### Running a Parameter Grid Exploration

`run_grid.py` runs many `run_coupled.py` (or `run_fast.py`) calls, one after another. The grid is defined in a YAML file in `grids/`: fixed parameters, lists of values to combine, the output directory and the run-name template. See `grids/README.md` for the keys.

```bash
python3 run_grid.py grids/n2dom.yaml --dry-run    # show the runs and their commands
python3 run_grid.py grids/n2dom.yaml
python3 run_grid.py grids/cia_comparison.yaml --only cia_pair=N2-CH4
```

| Grid file | Purpose |
|---|---|
| `grids/n2dom.yaml` | T_int × P_surf × manual elemental abundances, using `run_coupled.py`. |
| `grids/n2dom_fast.yaml` | The same kind of grid using the fast approximate RT (`run_fast.py`). |
| `grids/outgassing.yaml` | Outgassed atmospheres: T_int × atmodeller parameters (melt temperature, H ocean, C/H, N/C, fO2). |
| `grids/cia_comparison.yaml` | Every available CIA opacity source (see 2nd paper). Each source is swapped into HELIOS' `r50_kdistr` in turn, constant-chemistry runs are made, and the default is restored afterwards. `python3 run_grid.py --list-cia-sources [FOLDER]` / `--restore-cia-defaults [FOLDER]` list the sources and copy the default ones into `r50_kdistr`, from `$HELIOS_PATH/input/opacity/FOLDER` (default `hitran_cia`; e.g. `hitran_cia_new`, `hitran_cia_legacy`). A default missing from the folder is looked up without its option suffix (`N2-N2_2021_Tconst=50K` → `N2-N2_2021`); pairs without one keep their current file. |
| `grids/early_mars_cia.yaml` | Benchmark against Turbet+ (2020): 2 bar CO₂ with 80 % RH H₂O and trace H₂/CH₄, for different CO₂–H₂ / CO₂–CH₄ CIA sources. It needs `helios_inputs/param_EarlyMars.dat`. |
| `grids/cloud_test.yaml` | First cloud test: H₂O-ice deck (base 1, 0.1, 0.01 bar; τ = 0.1, 1, 10) + cloud-free baseline in an H₂-dominated exomoon atmosphere (see [Clouds](#clouds)). |

With `run_coupled.py`, runs that already finished are skipped, so an interrupted grid can simply be started again (an unfinished run restarts from scratch).

**Caution:** runs share working files, both in `ggchem_inputs/` and `helios_inputs/` and inside the HELIOS/GGchem installations. Simulations therefore must not run in parallel, and `run_grid.py` deliberately runs them sequentially. This includes starting several `run_coupled.py` processes by hand.

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

`cloud_test.py` summarises the cloud test (`grids/cloud_test.yaml`): table, T-P and mixing-ratio profiles, ΔT_surf against the cloud optical depth.

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
├─ run_grid.py                 # Runs a parameter grid defined in grids/*.yaml
├─ grids/                      # Grid files (N2dom, outgassing, CIA comparison, Early Mars, ...) + README
├─ get_last_ggchem.py          # Compute missing final GGchem output for finished/aborted runs
├─ analyze/                    # Analysis notebooks and tools
│  ├─ 1_IndividualRun.ipynb, 2_…, 3_…, A_…, B_…, C_cornerplots.ipynb
│  ├─ cia-opac_relevance.ipynb, opac_relevance.py
│  ├─ cloud_test.py            # Summary of the cloud test grid
│  └─ analyze_modules/         # data_loader.py (ChelioRun, sweep/matrix loaders), plot_utils.py
├─ chelio_sim/                 # Python package for simulation logic and utilities
│  ├─ external_runners.py      # Wrappers for HELIOS and GGchem
│  ├─ abundances.py            # Initial abundances (manual or atmodeller outgassing)
│  ├─ init_pt.py               # Initial P-T profiles
│  ├─ mixfile_utils.py         # GGchem → HELIOS mixfile, condensation limits, adiabatic-gradient table
│  ├─ rt_utils.py              # Approximate fast RT with on-the-fly Rosseland mean calculation*
│  ├─ species.py               # Active species of a run → HELIOS species file
│  ├─ grid.py                  # Grid expansion, CIA source swapping, warm start (used by run_grid.py)
│  ├─ clouds.py                # HELIOS Mie-file contract, size-averaged optics, tau <-> f_bot
│  ├─ make_mie_files.py        # Mie tables for HELIOS clouds with LX-MIE (+ --check)
│  └─ ...                      # Standalone helpers (escape, abundance and T-P conversion scripts)
├─ ggchem_inputs/
│  └─ param.in                 # GGchem's main parameter file (abundances.in, pt_helios.in are generated here)
├─ helios_inputs/
│  ├─ param.dat                # HELIOS's main parameter file (pre-set to Earth-sized moon around Jupiter-like FFP)
│  ├─ all_species.dat          # Catalog of species HELIOS has opacities for (P_BOA.dat is generated here)
│  └─ clouds/                  # refractive_index/ (tracked) and generated Mie tables, e.g. H2O_s/
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
