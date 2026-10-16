# Grid files

Each `*.yaml` here describes one parameter grid for `run_grid.py`:

```bash
python3 run_grid.py grids/n2dom.yaml --dry-run                 # list the runs and their commands
python3 run_grid.py grids/n2dom.yaml                           # run them, one after another
python3 run_grid.py grids/cia_comparison.yaml --only cia_pair=N2-CH4,CO2-CO2 --only internal_temp=145
python3 run_grid.py --list-cia-sources
python3 run_grid.py --restore-cia-defaults                     # after a killed CIA grid
```

| File | Grid |
|---|---|
| `h2dom.yaml` | H2-dominated exomoon atmospheres of the first paper: P_surf × T_int × (C+O) × C/O, plus nitrogen and Io-gravity variations |
| `n2dom.yaml` | T_int × P_surf × manual elemental abundances, equilibrium chemistry |
| `n2dom_fast.yaml` | the same with the fast RT (`run_fast.py`) |
| `outgassing.yaml` | T_int × atmodeller outgassing parameters (melt temperature, H ocean, C/H, N/C, fO2) |
| `cia_comparison.yaml` | every CIA source in `hitran_cia/`, 50/50 constant-chemistry atmospheres of the pair (2nd paper) |
| `early_mars_cia.yaml` | Early Mars benchmark vs. Turbet+ (2020) for CO2-H2 / CO2-CH4 CIA sources |

To make a new grid, copy the closest file. Paths are relative to the Chelio root.

## Keys

| Key | Meaning |
|---|---|
| `runner` | `coupled` (`run_coupled.py`, default) or `fast` (`run_fast.py`) |
| `out_dir` | root output directory, one subdirectory per run |
| `name` | run name, a Python format string over the run parameters, e.g. `"Tint={internal_temp}K_{trace_pct:g}pct"` |
| `config` | base config file (default: `config.yaml`) |
| `helios_param` | HELIOS param file instead of `helios_inputs/param.dat` (coupled only). Single runs can override it with the parameter `helios_param_file`, e.g. in a `cases` block |
| `fixed` | parameters shared by all runs |
| `grid` | parameter → list of values; every combination is run (first key varies slowest) |
| `cases` | list of parameter blocks, each one crossed with `grid`. List values inside a block are crossed within the block |
| `derive` | hooks from `chelio_sim/grid.py` (`DERIVE_HOOKS`) that compute more parameters per run: `cia_pair_mixing`, `early_mars_mixing`, `c_plus_o_abundances` |
| `warm_start` | parameter (e.g. `internal_temp`) that must appear in `name`. Each run starts from the last T-P profile of the existing run in `out_dir` with the next-higher value, and higher values are run first (coupled only) |

A parameter that the runner accepts as a CLI flag (`internal_temp`, `surface_pressure`, `a_h`, `chemistry`, `constant_mixing_ratios`, `relative_humidity`, `active_species`, …) is passed as `--key value`. All other parameters are **labels**: they are only used in `name`, in `--only` filters and by hooks. The header printed by `run_grid.py` lists them, so check it for typos. `constant_mixing_ratios` can be written as a mapping (`{N2: 0.5, CH4: 0.5}`). `active_species` (default `auto`, see the main README) is written comma-separated, e.g. `active_species: "H2O,CO2,CH4,H2"`, since a YAML list would be read as several values to run.

Available in `name` in addition to the parameters: `psurf_bar` (= int(surface_pressure / 1e6)) and, for CIA grids, `cia_pair`.

**CIA sources:** the key `cia` takes a source from `$HELIOS_PATH/input/opacity/hitran_cia/` (file name without `.h5`), or `auto` for all of them. The runs of one source are grouped. The source is copied into `r50_kdistr/CIA_<pair>_opac_ip_kdistr.h5` before them and replaced by the default (`DEFAULT_CIA_SOURCES` in `chelio_sim/grid.py`) afterwards, also if the grid is interrupted. In CIA grids, `surface_pressure` values may also be `critical`: the lower critical pressure of the pair's two molecules (numbers are used as they are, e.g. `[1.0e6, critical]`).

**Output:** runs that already finished are reported as skipped. A copy of the grid file is written to `<out_dir>/grid_<file>.yaml`, and the output of failed runs to `<out_dir>/grid_failures.log`.
