# Chelio Project Guidelines

## Overview
Chelio couples HELIOS (1D radiative transfer) with GGchem (equilibrium chemistry) for self-consistent atmospheric simulations. All scripts run from the repository root.

## Architecture

### Core Components
- **run_coupled.py**: Main entry point for all simulations. Orchestrates HELIOS-GGchem coupling loops.
- **chelio_sim/**: Python package containing simulation utilities
  - `external_runners.py`: Subprocess wrappers for HELIOS and GGchem executables
  - `mixfile_utils.py`: Converts between GGchem output and HELIOS input formats. Handles condensation, VMR normalization, and delad/kappa table generation.
  - `abundances.py`: Calculates initial elemental abundances (via atmodeller or manual input)
  - `init_pt.py`: Generates initial pressure-temperature profiles
- **config.yaml**: Central configuration for all simulation parameters

### Data Flow
1. Calculate initial abundances → Run GGchem → Convert to HELIOS mixfile
2. Run HELIOS with mixfile → Get new T-P profile → Update GGchem input
3. Iterate until convergence (coupling loop, iterations i_min to i_max)

### Chemistry Modes
- **ggchem**: Full equilibrium chemistry via GGchem (default)
- **constant**: Fixed mixing ratios with condensation limits applied

## Code Style

### Python Conventions
- Python 3.x required
- Use `logging` module (not print statements) - see run_coupled.py:16-48 for setup pattern
- NumPy/SciPy for numerical operations
- Type hints preferred but not required
- Subprocess execution via `external_runners.run_command()` for live output streaming

### File I/O Patterns
- Use `shutil.copy()` for file operations between directories
- Always use `os.path.join()` for cross-platform paths
- Config paths support environment variable expansion: `${GGCHEM_PATH}`, `${HELIOS_PATH}`

## Build and Test

### Installation
```bash
pip install -r requirements.txt
```

### Required Environment Variables
```bash
export GGCHEM_PATH="/path/to/ggchem"
export HELIOS_PATH="/path/to/helios"
```
Alternatively, edit paths directly in config.yaml.

### Running Simulations
```bash
# From repository root only
python3 run_coupled.py --name "RunName"

# Override config parameters
python3 run_coupled.py --name "Test" --internal_temp 300 --surface_pressure 1e8

# Grid exploration (bash loop)
for T in 100 200 300; do
  python3 run_coupled.py --name "Grid_T${T}" --internal_temp ${T}
done
```

### Output Structure
- All outputs saved to `output/<run_name>/`
- Coupling T-P profiles: `<name>_tp_coupling_{i}.dat`
- Vertical mixing ratios: `vertical_mix_{i}.dat`
- GGchem snapshots: `Static_Conc_{i}.dat`
- Log file: `run.log`

## Project Conventions

### Adding New Parameters
1. Add to `simulation_params` section in config.yaml
2. Add argparse argument in run_coupled.py (lines 98-119)
3. Apply override in main() (lines 132-134)

### Extending Chemistry Modes
- Implement new mode in run_coupled.py coupling loop (lines 327-442)
- Follow pattern: chemistry_mode == "your_mode"
- Must generate HELIOS-compatible mixfile at each iteration

### External Process Integration
- All external commands via `external_runners.run_command()`
- Logs stdout/stderr live via logging module
- Raises `subprocess.CalledProcessError` on non-zero exit

### File Naming Convention
- Input templates: `helios_inputs/`, `ggchem_inputs/`
- Iteration-specific outputs include iteration number: `_tp_coupling_{i}.dat`, `vertical_mix_{i+1}.dat`

## Integration Points

### HELIOS Dependencies
- Requires HELIOS Python modules in `${HELIOS_PATH}/source/`
- Imports `species_database.species_lib` for molecular weights
- Reads `param.dat` from helios_inputs/ and copies to HELIOS_PATH before runs

### GGchem Dependencies
- Expects `./ggchem` executable in `${GGCHEM_PATH}`
- Input files: `abund_helios.in`, `structures/pt_helios.in`, `input/param_helios.in`
- Output file: `Static_Conc.dat`
- JANAF thermodynamic data in `${GGCHEM_PATH}/data/JANAF/{species}.txt` used for heat capacity calculations

### Analysis Tools
- Jupyter notebooks in `analyze/` directory
- Use `analyze_modules.data_loader` and `analyze_modules.plot_utils` for loading/plotting results

## Security

### Subprocess Execution
- **Never** pass unsanitized user input directly to subprocess commands
- Use list-form commands (not shell=True) in `external_runners.run_command()`
- All external executables must be in trusted installation directories

### File Path Handling
- Environment variable expansion uses `os.environ.get()` with validation
- Check directory existence before running external codes (run_coupled.py:157-165)
- Use `os.makedirs(exist_ok=True)` to avoid race conditions

## Common Pitfalls

1. **Running from wrong directory**: Always execute scripts from `/chelio/` root, not subdirectories
2. **Missing environment variables**: Script exits with clear error if GGCHEM_PATH or HELIOS_PATH unset
3. **Pressure grid spacing**: HELIOS kappa/delad tables require constant log10(P) spacing - validated in `mixfile_utils._validate_log10_pressure_grid()`
4. **VMR normalization**: Mixing ratios must sum to 1.0 at each layer - both ggchem and constant modes handle this
5. **Condensation limits**: Species VMRs limited by saturation pressure from `mixfile_utils.p_sat()`
