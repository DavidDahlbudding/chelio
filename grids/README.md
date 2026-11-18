# Grid files

Each `*.yaml` here describes one parameter grid for `run_grid.py`:

```bash
python3 run_grid.py grids/n2dom.yaml --dry-run                 # list the runs and their commands
python3 run_grid.py grids/n2dom.yaml                           # run them, one after another
python3 run_grid.py grids/cia_comparison.yaml --only cia_pair=N2-CH4,CO2-CO2 --only internal_temp=145
python3 run_grid.py --list-cia-sources [hitran_cia_new]        # default folder: hitran_cia
python3 run_grid.py --restore-cia-defaults                     # after a killed CIA grid
python3 run_grid.py --restore-cia-defaults hitran_cia_new      # install the defaults of another source folder
```

| File | Grid |
|---|---|
| `h2dom.yaml` | H2-dominated exomoon atmospheres of the first paper: P_surf × T_int × (C+O) × C/O, plus nitrogen and Io-gravity variations |
| `h2dom_constant_session.yaml` | fast reproduction of its main grid (Earth, no N) with the session runner: constant CH4/H2O/H2 from X_C, X_O + cold trap, constant kappa, only H2-H2 CIA (`test_slurm/h2dom_constant_session.sbatch`, `analyze/compare_h2dom_reproduction.py`) |
| `n2dom.yaml` | T_int × P_surf × manual elemental abundances, equilibrium chemistry |
| `n2dom_fast.yaml` | the same with the fast RT (`run_fast.py`) |
| `outgassing.yaml` | T_int × atmodeller outgassing parameters (melt temperature, H ocean, C/H, N/C, fO2) |
| `cia_comparison.yaml` | every CIA source in `hitran_cia/`, 50/50 constant-chemistry atmospheres of the pair (2nd paper) |
| `early_mars_cia.yaml` | Early Mars benchmark vs. Turbet+ (2020) for CO2-H2 / CO2-CH4 CIA sources |
| `cloud_test.yaml` | first cloud test: H2O-ice deck (base 1, 0.1, 0.01 bar × τ = 0.1, 1, 10) + cloud-free baseline, H2-dominated exomoon |

To make a new grid, copy the closest file. Paths are relative to the Chelio root.

## Keys

| Key | Meaning |
|---|---|
| `runner` | `coupled` (`run_coupled.py`, default), `fast` (`run_fast.py`) or `session` (one HELIOS run per grid point with the composition from a pre-computed chemistry grid, all runs in one in-process HELIOS session; see below) |
| `out_dir` | root output directory, one subdirectory per run |
| `name` | run name, a Python format string over the run parameters, e.g. `"Tint={internal_temp}K_{trace_pct:g}pct"` |
| `config` | base config file (default: `config.yaml`) |
| `helios_param` | HELIOS param file instead of `helios_inputs/param.dat` (coupled only). Single runs can override it with the parameter `helios_param_file`, e.g. in a `cases` block |
| `fixed` | parameters shared by all runs |
| `grid` | parameter → list of values; every combination is run (first key varies slowest) |
| `cases` | list of parameter blocks, each one crossed with `grid`. List values inside a block are crossed within the block |
| `derive` | hooks from `chelio_sim/grid.py` (`DERIVE_HOOKS`) that compute more parameters per run: `cia_pair_mixing`, `early_mars_mixing`, `c_plus_o_abundances`, `c_plus_o_constant_mixing` (same labels, constant_mixing_ratios: X_CH4 = X_C, X_H2O = X_O, X_H2 = (X_H − 4 X_C − 2 X_O)/2, normalised), `cloud_from_tau` |
| `warm_start` | parameter (e.g. `internal_temp`) that must appear in `name`. Each run starts from the last T-P profile of the existing run in `out_dir` with the next-higher value, and higher values are run first (coupled and session). Session runner only: also `previous` (last finished run of this invocation), `nearest` (finished run of this grid closest in the numeric run parameters) or `none` |
| `session` | session runner settings (defaults in `chelio_sim/session_runner.py: SESSION_DEFAULTS`): `chem_dT`, `chem_t_min`, `chem_t_max`, `chem_p_spacing` (`opacity` = 1/3 dex, `layers` = HELIOS layer + interface pressures, or a number in dex), `chem_eqcond`, `max_iter`, `rad_max_iter`, `cold_start_temperature`, `min_conv_iter`, `stitch_flip_limit`, `stitch_flip_start`, `stitch_after_iter`, `stitch_surface`, `write_tp_every`, `tp_coupling_every` (`<name>_tp_coupling_<i>.dat` every n iterations, i = radiation + convection iterations, final = highest i; default 100, 0 = final only), `warm_start`, `init_from` (directory with runs of the same names to start from), and the column model: `chemistry_source`, `vmr_callable`, `vmr_callable_kwargs`, `condensibles`, `convection`, `surface_pressure_model`, `composition_damping`, `vapour_supply` (see *Column model* below) |

A parameter that the runner accepts as a CLI flag (`internal_temp`, `surface_pressure`, `a_h`, `chemistry`, `constant_mixing_ratios`, `relative_humidity`, `active_species`, …) is passed as `--key value`. All other parameters are **labels**: they are only used in `name`, in `--only` filters and by hooks. The header printed by `run_grid.py` lists them, so check it for typos. `constant_mixing_ratios` can be written as a mapping (`{N2: 0.5, CH4: 0.5}`). `active_species` (default `auto`, see the main README) is written comma-separated, e.g. `active_species: "H2O,CO2,CH4,H2"`, since a YAML list would be read as several values to run.

**Clouds** (coupled only): the cloud keys of `config.yaml` (`cloud_mie_path`, `cloud_r_mode`, `cloud_r_std_dev`, `cloud_bottom_pressure`, `cloud_bottom_mixing_ratio`, `cloud_to_gas_scale_height_ratio`) are runner flags. With `derive: [cloud_from_tau]`, set the labels `cloud_tau` (total optical depth, Planck mean at `internal_temp`; 0 = cloud-free) and `cloud_pbot_bar` (cloud base in bar) instead of `cloud_bottom_pressure` and `cloud_bottom_mixing_ratio`; the hook also provides the name field `cloud_name` (`none` or e.g. `H2Os_pbot=1e-1_tau=1`). See `cloud_test.yaml` and the main README.

Available in `name` in addition to the parameters: `psurf_bar` (= int(surface_pressure / 1e6)) and, for CIA grids, `cia_pair`.

**CIA sources:** the key `cia` takes a source from `$HELIOS_PATH/input/opacity/hitran_cia/` (file name without `.h5`), or `auto` for all of them. The runs of one source are grouped. The source is copied into `r50_kdistr/CIA_<pair>_opac_ip_kdistr.h5` before them and replaced by the default (`DEFAULT_CIA_SOURCES` in `chelio_sim/grid.py`) afterwards, also if the grid is interrupted. In CIA grids, `surface_pressure` values may also be `critical`: the lower critical pressure of the pair's two molecules (numbers are used as they are, e.g. `[1.0e6, critical]`).

**Output:** runs that already finished are reported as skipped. A copy of the grid file is written to `<out_dir>/grid_<file>.yaml`, and the output of failed runs to `<out_dir>/grid_failures.log`.

**CIA selection** (session runner only): the run parameter `cia_pairs` (`auto` = default: every pair whose partners are active; `none`; or a comma-separated list such as `H2-H2,H2-CH4`) limits the CIA pairs written to the species file.

**Session runner** (`runner: session`): `run_grid.py` keeps one `HeliosSession` (`chelio_sim/helios_session.py`) for all runs, so HELIOS is imported, its kernels compiled and the opacity tables read once (and kept on the GPU if they fit). Each run is a single HELIOS run whose composition follows the temperature during the run, from a chemistry grid (`chelio_sim/chem_grid.py`): GGchem on a (T, P) grid with equilibrium condensation but **without rain-out** (`remove_condensates .false.`), or the saturation-capped constant mixing ratios of `chemistry: constant`. Grids are cached in `<out_dir>/chem_grids/` and reused by all runs with the same composition. Per run: `<name>_tp_coupling_0.dat` (final T-P), `<name>_restart.npz` (exact state for warm starts), `<name>_session.json` (convergence, iterations, timings, stitched layers, how often the profile left the chemistry grid), `vertical_mix_0.dat`, plus the HELIOS output. Finished runs (restart file + convergence flag 1) are skipped. `grids/session_tests/` + `slurm/session_tests.sbatch` compare it with the legacy coupling (`analyze/compare_session_tests.py`).

`run_grid.py` writes `<out_dir>/grid_timings.csv` (wall time per run) for all runners.

## Column model (session runner)

The session runner can replace HELIOS' dry, table-driven convection and the fixed chemistry grid by a **column model** (`chelio_sim/column.py`). It is switched on as soon as one of the following `session` settings differs from its default. Each of them can also be set per run in `grid` or `cases`; dict values (e.g. `convection: {inhibition: false}`) are then merged into the dict of the `session` block.

| Setting | Meaning |
|---|---|
| `chemistry_source` | `grid` (default: the chemistry grid above), `constant_exact` (`constant_mixing_ratios` without grid and without local caps; the condensibles are capped by the cold trap) or `callable` |
| `vmr_callable`, `vmr_callable_kwargs` | for `callable`: `"package.module:name"`, a function `f(p [dyn/cm²], T [K], species) -> {species: VMR array}` (with kwargs: `name(**kwargs)` returns it) |
| `condensibles` | (H₂O: p_sat and latent heat blend smoothly from ice to liquid over 250–273.16 K, `thermo.H2O_MIXED_PHASE`, so the moist adiabat has no kink at the triple point) `[]` (default), `auto` (active species among H2O, CO2, CH4, NH3, N2, CO, O2) or a list / comma-separated string, e.g. `[H2O, NH3]`. Cold trap: each condensible is capped by the running minimum (bottom to top) of RH·p_sat/p, the gas removed goes to the other species. `relative_humidity` applies to H2O **Not recommended with `chemistry: equilibrium` (GGchem grid) for now** (warning in the log; see `vapour_supply`) |
| `convection` | dict (`chelio_sim/convection.py: ConvectionSettings`): `criterion` (`schwarzschild` default, `ledoux`), `moist` (true: multi-condensible pseudoadiabat where saturated), `inhibition` (**experimental**, true by default: Leconte et al. 2017, moist convection is inhibited where saturated with Γ > 1, i.e. q > q_cri; logged as a warning when active, not validated yet), `inhibition_mode` (`jump`, default: the inhibited radiative layer is sub-grid, as Leconte et al. find it thin and almost isobaric; a saturated parcel with Γ > 1 relaxes to Γ = 1 over `jump_width` (0.1 in ln p), so the targets contain the temperature jump and the deep and moist zones form one convective zone; `radiative`: saturated layers with Γ > 1 are resolved as stable radiative layers that are never stitched, which only works if a grid layer can stay inside the narrow temperature band q_cri < q_sat < q_deep, usually not on HELIOS' 0.1 dex grid), `semiconvection` (`onset_only`: Ledoux decides the onset, pairs that are already convective keep the adiabat test; `strict`), `multi` (`sum`, `dominant`), `thermo_source` (`host`: c_p and adiabats from the column model; `table`: HELIOS' kappa table, dry only, for regression tests), `inhibition_hysteresis` (0.02), `moist_tolerance` (1e-4, 1e-3 before 2026-11-17, which let moist zones overshoot and leave single cold layers at their top: a convective pair with a moist target stays convective unless its upper layer is more than this fraction warmer than the target allows; HELIOS' ~1e-6 is kept for dry pairs, but moist targets depend strongly on T, most of all at the lifting condensation level, and moist zones would fall apart), `unfreeze` (true: a layer radiative for one adaptive interval whose HELIOS time-step prefactor has decayed below 1e-5 while it was convective gets 1e-3 again), `max_adjust_iter` (1000), `parcel_substeps` (2), `kappa` (None; a number, e.g. 0.285714, gives a constant dry adiabat with c_p = R/κ like HELIOS' numeric `kappa value`; needs `moist: false`, `inhibition: false`, `thermo_source: host`) |
| `surface_pressure_model` | dict (`chelio_sim/surface.py`): `callable` (`"package.module:name"`, `f(ColumnState) -> p_surf` [dyn/cm²]), `kwargs`, `update_every` (200 iterations), `relax` (0.5, in log p), `tol` (1e-3), `p_min`, `p_max` (also the top of the chemistry grid), `max_factor` (2 per update), `settle`. The run parameter `surface_pressure` is the start value; the grid is rebuilt during the run (same number of layers and TOA). Examples: `chelio_sim.surface:fixed_inventory` (`species`, `p_inventory` [dyn/cm²]), `chelio_sim.surface:ocean_plus_inventory` (`condensible`, `p_inventory` [dyn/cm²], `rh`, optional `p_ocean` [dyn/cm²]: finite ocean, the vapour pressure stays at `p_ocean` once RH·p_sat exceeds it, also above the critical point where p_sat = ∞) |
| `composition_damping` | 0 ≤ w < 1: the log VMRs handed to HELIOS are relaxed with the previous update |
| `vapour_supply` | vapour of the condensibles available from below: `running_max` (default; a decrease of a condensible's base VMR with height counts as condensation and is redone by the cold trap with one saturation curve, needed for equilibrium grids, which condense locally and whose interpolated VMRs lie below saturation), `saturated` (carry only where the base VMR is saturated within 2%; keeps chemical decreases), `base` Known problem, not fixed yet: the interpolated grid VMRs of condensibles lie far below saturation at low T (orders of magnitude at ≈80 K with 50 K steps), see `docs/column_model_handoff.md` |

With the column model HELIOS gets c_p and the adiabatic exponents from chelio (no kappa table, which also avoids the log-T axis of HELIOS' `cp_interpol` kernel), and every run writes `<name>_column.dat` (per layer: μ, c_p, ∇_ad, target exponents of the lower/upper half layer, onset thresholds, ∇μ, Γ, inhibited and convective flags, VMR and saturation ratio of each condensible). `<name>_session.json` gets `p_surf` and `column` (history of p_surf and of the numbers of inhibited / saturated layers, regrids, final inhibited and saturated layers). The column-model files use HELIOS' cgs units (p in dyn/cm², c_p in erg mol⁻¹ K⁻¹); the T-P files stay in bar as before. Test grids: `grids/column_tests/` (`c1` H2O inhibition in H2, `c2` nondilute steam in N2, `c3` variable surface pressure, `c4` two condensibles, `c5` equilibrium grid + cold traps + Ledoux) and `grids/session_tests/t6*` (regression without new physics).
