"""
Parameter grids for run_grid.py: expands a grid file (grids/*.yaml) into single runs of
run_coupled.py / run_fast.py and provides the helpers to run them one after another.

Grid file keys (see grids/README.md for the full description and grids/*.yaml for examples):

  runner        "coupled" (run_coupled.py, default) or "fast" (run_fast.py)
  out_dir       root output directory of the runs
  name          run name template, a Python format string over the run parameters
  config        optional base config file (default: the runner's config.yaml)
  helios_param  optional HELIOS param file (run_coupled.py --helios_param_file); single runs can
                override it with the parameter `helios_param_file`
  fixed         parameters shared by every run
  grid          parameter -> list of values, all combinations are run
  cases         optional list of parameter blocks, each one crossed with `grid`; list values
                inside a block are crossed as well
  derive        optional list of hook names (DERIVE_HOOKS) that compute further parameters
  warm_start    optional parameter name: start each run from the last T-P profile of the existing
                run in out_dir that has the next-higher value of this parameter (rest of name equal)

Parameters that the runner accepts as a CLI flag are passed as --<key> <value>, all others are
labels that are only used in the name, for --only filters and by derive hooks.
The key `cia` is reserved: its value is a CIA source in $HELIOS_PATH/input/opacity/hitran_cia/
(file name without .h5, or "auto" for all of them), which is copied into r50_kdistr while the run
is going and replaced by the default source afterwards. With a `cia` source, surface_pressure may
also be "critical": the lower critical pressure of the two molecules of the pair.
"""
import contextlib
import glob
import itertools
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from chelio_sim import run_modes

CHELIO_ROOT = Path(__file__).resolve().parent.parent

RUNNER_SCRIPTS = {"coupled": "run_coupled.py", "fast": "run_fast.py"}
_MODE_KEYS = ("chemistry", "outgas_or_manual", "uncoupled_outgassed_run", "active_species")
RUNNER_FLAGS = {
    "coupled": {*run_modes.SIM_PARAMS, *_MODE_KEYS, "constant_mixing_ratios", "relative_humidity", "init_pt_file", "helios_param_file"},
    "fast": {*run_modes.SIM_PARAMS, *_MODE_KEYS, "constant_mixing_ratios"},
}
SPEC_KEYS = {"runner", "out_dir", "name", "config", "helios_param", "fixed", "grid", "cases", "derive", "warm_start"}

# Default CIA sources (from create_all_CIAopacs.bash), restored in r50_kdistr after testing other sources
DEFAULT_CIA_SOURCES = {
    "CH4-CH4": "CH4-CH4_2011",
    "CO2-CH4": "CO2-CH4_2024_main",
    "CO2-CO2": "CO2-CO2_2024",
    "CO2-H2": "CO2-H2_2024",
    "H2-CH4": "H2-CH4_eq_2011",
    "N2-CH4": "N2-CH4_2024_Tconst=50K",
    "N2-H2": "N2-H2_2024",
    "N2-H2O": "N2-H2O_2018",
    "N2-N2": "N2-N2_2021_Tconst=50K",
    "H2-H2": "H2-H2_2018+02",
}


@dataclass
class Task:
    name: str
    params: dict


# --- Loading ---

class _GridLoader(yaml.SafeLoader):
    """SafeLoader that also reads 1e7 / 1.0e7 (no exponent sign) as floats, like YAML 1.2 does."""


_GridLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)[eE][-+]?\d+$"),
    list("-+0123456789."),
)


def load_grid(path):
    """Reads and checks a grid file."""
    with open(path) as f:
        spec = yaml.load(f, Loader=_GridLoader)
    if not isinstance(spec, dict):
        raise ValueError(f"grid file {path} does not contain a mapping")

    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown grid file keys {sorted(unknown)}; allowed: {sorted(SPEC_KEYS)}")
    for key in ("out_dir", "name"):
        if not spec.get(key):
            raise ValueError(f"grid file needs `{key}`")
    spec.setdefault("runner", "coupled")
    if spec["runner"] not in RUNNER_SCRIPTS:
        raise ValueError(f"runner must be one of {list(RUNNER_SCRIPTS)}, not {spec['runner']!r}")
    spec["fixed"] = spec.get("fixed") or {}
    spec["grid"] = spec.get("grid") or {}
    spec["cases"] = spec.get("cases") or [{}]
    spec["derive"] = spec.get("derive") or []

    for key, value in spec["fixed"].items():
        if isinstance(value, list):
            raise ValueError(f"fixed.{key} is a list; put parameters with several values under `grid`")
    for hook in spec["derive"]:
        if hook not in DERIVE_HOOKS:
            raise ValueError(f"unknown derive hook {hook!r}; available: {sorted(DERIVE_HOOKS)}")
    for case in spec["cases"]:
        both = set(case) & set(spec["grid"])
        if both:
            raise ValueError(f"{sorted(both)} set in both `grid` and `cases`")
    if spec["runner"] != "coupled":
        if spec.get("helios_param"):
            raise ValueError("helios_param only works with runner: coupled")
        if spec.get("warm_start"):
            raise ValueError("warm_start only works with runner: coupled (run_fast.py has no --init_pt_file)")
    return spec


# --- Expanding the grid into runs ---

def _combinations(block):
    """All combinations of the list values in `block` (first key varies slowest)."""
    keys = list(block)
    values = []
    for key in keys:
        value = block[key]
        if key == "cia" and value == "auto":
            value = discover_cia_sources()
        values.append(value if isinstance(value, list) else [value])
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def format_name(template, params):
    try:
        return template.format(**params)
    except KeyError as e:
        raise ValueError(f"name template uses {e}, which is not a parameter of the run (known: {sorted(params)})") from None


def expand_tasks(spec):
    """Expands fixed x cases x grid into Tasks, in run order (cases outermost, then grid)."""
    runner = spec["runner"]
    other_flags = set().union(*RUNNER_FLAGS.values()) - RUNNER_FLAGS[runner]
    tasks = []
    for case in spec["cases"]:
        for case_params in _combinations(case):
            for grid_params in _combinations(spec["grid"]):
                params = {**spec["fixed"], **grid_params, **case_params}
                if "cia" in params:
                    params["cia_pair"] = cia_pair(params["cia"])
                if params.get("surface_pressure") == "critical":
                    params["surface_pressure"] = critical_surface_pressure(params)
                for hook in spec["derive"]:
                    DERIVE_HOOKS[hook](params)
                if isinstance(params.get("surface_pressure"), (int, float)):
                    params["psurf_bar"] = int(params["surface_pressure"] / 1e6)
                unsupported = set(params) & other_flags
                if unsupported:
                    raise ValueError(f"{sorted(unsupported)} not supported by {RUNNER_SCRIPTS[runner]}")
                tasks.append(Task(format_name(spec["name"], params), params))

    names = [task.name for task in tasks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"name template gives the same name to several runs, e.g. {duplicates[0]!r}")
    return tasks


def label_keys(spec, tasks):
    """Parameters that are not passed to the runner (only used in names, filters and hooks)."""
    keys = {key for task in tasks for key in task.params}
    return sorted(keys - RUNNER_FLAGS[spec["runner"]])


def _matches(value, wanted):
    if str(value) == wanted:
        return True
    try:
        return float(value) == float(wanted)
    except (TypeError, ValueError):
        return False


def filter_tasks(tasks, only):
    """Keeps tasks matching all `key=value[,value...]` filters (several values of one key: any of them)."""
    all_tasks = tasks
    for item in only or []:
        key, sep, values = item.partition("=")
        if not sep:
            raise ValueError(f"--only expects key=value, got {item!r}")
        available = [task.params[key] for task in all_tasks if key in task.params]
        if not available:
            raise ValueError(f"--only: no run has a parameter {key!r}")
        wanted = values.split(",")
        for w in wanted:
            if not any(_matches(value, w) for value in available):
                choices = ", ".join(dict.fromkeys(str(value) for value in available))
                raise ValueError(f"--only: {key}={w} is not in the grid; available: {choices}")
        tasks = [task for task in tasks if any(_matches(task.params.get(key), w) for w in wanted)]
    return tasks


def order_tasks(spec, tasks):
    """Groups runs by CIA source (one swap per source) and, with warm_start, runs higher values first."""
    groups = {}
    for task in tasks:
        groups.setdefault(task.params.get("cia"), []).append(task)
    axis = spec.get("warm_start")
    if axis:
        for group in groups.values():
            group.sort(key=lambda task: -float(task.params[axis]))
    return [task for group in groups.values() for task in group]


# --- Running ---

def build_command(spec, task, init_pt_file=None):
    runner = spec["runner"]
    cmd = [sys.executable, RUNNER_SCRIPTS[runner], "--name", task.name, "--out_dir", spec["out_dir"]]
    if spec.get("config"):
        cmd += ["--config", spec["config"]]
    if spec.get("helios_param") and "helios_param_file" not in task.params:
        cmd += ["--helios_param_file", spec["helios_param"]]
    for key, value in task.params.items():
        if key not in RUNNER_FLAGS[runner]:
            continue
        if key == "constant_mixing_ratios" and isinstance(value, dict):
            value = ",".join(f"{species}={vmr}" for species, vmr in value.items())
        cmd += [f"--{key}", str(value)]
    if init_pt_file:
        cmd += ["--init_pt_file", init_pt_file]
    return cmd


def run_task(cmd, verbose=False):
    """Runs one simulation from the Chelio root. Returns ("done" | "skipped" | "failed", result)."""
    result = subprocess.run(cmd, cwd=CHELIO_ROOT, text=True, capture_output=not verbose)
    if result.returncode == 0:
        return "done", result
    if result.returncode == run_modes.EXIT_ALREADY_DONE:
        return "skipped", result
    return "failed", result


class _Wildcard:
    """Stands in for the warm-start parameter when turning the name template into a regex."""
    token = "@@WARM_START@@"

    def __format__(self, format_spec):
        return self.token


def check_warm_start(spec, task):
    axis = spec["warm_start"]
    if axis not in task.params:
        raise ValueError(f"warm_start parameter {axis!r} is not a parameter of the runs")
    if _Wildcard.token not in format_name(spec["name"], {**task.params, axis: _Wildcard()}):
        raise ValueError(f"warm_start parameter {axis!r} must appear in the name template")


def find_warm_start(spec, task):
    """
    Last T-P profile ({run}_tp_coupling_{i}.dat with the highest i) of the existing run in out_dir
    whose name only differs in the warm_start parameter, with the lowest value above this run's.
    Returns None if there is none.
    """
    axis = spec["warm_start"]
    value = float(task.params[axis])
    template = format_name(spec["name"], {**task.params, axis: _Wildcard()})
    number = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    pattern = re.compile(number.join(re.escape(part) for part in template.split(_Wildcard.token)))

    out_dir = CHELIO_ROOT / spec["out_dir"]
    best_value, best_run = float("inf"), None
    for entry in os.listdir(out_dir) if out_dir.is_dir() else []:
        match = pattern.fullmatch(entry)
        if match and (out_dir / entry).is_dir() and value < float(match.group(1)) < best_value:
            best_value, best_run = float(match.group(1)), entry
    if best_run is None:
        return None

    tp_pattern = re.compile(rf"{re.escape(best_run)}_tp_coupling_(\d+)\.dat")
    tp_files = [(int(m.group(1)), f) for f in os.listdir(out_dir / best_run) if (m := tp_pattern.fullmatch(f))]
    if not tp_files:
        return None
    return str(out_dir / best_run / max(tp_files)[1])


def critical_surface_pressure(params):
    """surface_pressure: critical -> the lower critical pressure [dyn/cm^2] of the two molecules of the run's CIA pair."""
    if "cia_pair" not in params:
        raise ValueError("surface_pressure: critical needs a `cia` source (the pressure is taken from its CIA pair)")
    from chelio_sim.mixfile_utils import mol_dict
    return float(min(mol_dict[mol]["critical"][1] for mol in params["cia_pair"].split("-")))


# --- CIA sources ---

def _opacity_dir(sub):
    try:
        return os.path.join(os.environ["HELIOS_PATH"], "input", "opacity", sub)
    except KeyError:
        raise RuntimeError("HELIOS_PATH is not set (needed to swap CIA sources)") from None


def cia_pair(source):
    """'CO2-CH4_2024_main' -> 'CO2-CH4'"""
    return source.split("_")[0]


def cia_target_path(pair):
    """'N2-CH4' -> $HELIOS_PATH/input/opacity/r50_kdistr/CIA_N2CH4_opac_ip_kdistr.h5"""
    return os.path.join(_opacity_dir("r50_kdistr"), f"CIA_{pair.replace('-', '')}_opac_ip_kdistr.h5")


def discover_cia_sources():
    """All CIA sources (file names without .h5) in hitran_cia/, sorted."""
    return sorted(Path(f).stem for f in glob.glob(os.path.join(_opacity_dir("hitran_cia"), "*.h5")))


def install_cia_source(source):
    source_path = os.path.join(_opacity_dir("hitran_cia"), f"{source}.h5")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"CIA source {source_path} not found")
    target = cia_target_path(cia_pair(source))
    print(f"CIA: {source}.h5 -> {os.path.basename(target)}")
    shutil.copy2(source_path, target)


def restore_default_cia(pairs=None):
    """Copies the default source of each pair (default: all pairs in DEFAULT_CIA_SOURCES) into r50_kdistr."""
    for pair in pairs or DEFAULT_CIA_SOURCES:
        default = DEFAULT_CIA_SOURCES.get(pair)
        if default is None:
            print(f"Warning: no default CIA source defined for {pair}; r50_kdistr keeps the tested one")
        elif not os.path.exists(os.path.join(_opacity_dir("hitran_cia"), f"{default}.h5")):
            print(f"Warning: default CIA source {default}.h5 not found")
        else:
            install_cia_source(default)


@contextlib.contextmanager
def swapped_cia(source):
    """Installs a CIA source for the duration of the block and restores the pair's default afterwards."""
    pair = cia_pair(source)
    install_cia_source(source)
    try:
        yield
    finally:
        if source != DEFAULT_CIA_SOURCES.get(pair):
            restore_default_cia([pair])


# --- Derive hooks: fn(params) that adds/changes parameters of one run in place ---

def cia_pair_mixing(params):
    """Constant mixing ratios for the CIA pair: 50/50 for cross-CIA, 100 % for self-CIA."""
    species = {"N2": 0.0, "CH4": 0.0, "CO2": 0.0, "H2": 0.0, "H2O": 1e-30}
    mol1, mol2 = params["cia_pair"].split("-")
    if mol1 == mol2:
        species[mol1] = 1.0
    else:
        species[mol1] = 0.5
        species[mol2] = 0.5
    params["constant_mixing_ratios"] = species


def early_mars_mixing(params):
    """
    CO2 background with trace_pct % of trace_gas (H2 or CH4) and H2O (capped at relative_humidity * p_sat/P).
    The other one of H2/CH4 is set to a negligible floor value.
    """
    trace_vmr = params["trace_pct"] / 100.0
    non_trace = "CH4" if params["trace_gas"] == "H2" else "H2"
    params["constant_mixing_ratios"] = {"CO2": 1.0 - trace_vmr, "H2O": 1.0, params["trace_gas"]: trace_vmr, non_trace: 1e-30}


def _format_e_num(num):
    """1e6 -> '1e6', 3.16e-3 -> '3.16e-3' (as `_format_e_nums` in analyze_modules/data_loader.py, keep in sync)."""
    text = f"{num:.2e}".replace("0", "").replace(".e", "e").replace("+", "")
    return text + "0" if text.endswith("e") else text


def c_plus_o_abundances(params):
    """
    H2-dominated atmospheres of the FFP exomoon paper: elemental abundances a_h/a_c/a_o/a_n from the labels
    CplusO = (C+O)/(H+C+O), CtoO = C/O and X_N (N fraction of all atoms, default 0), as the former calc_abundances.py.
    Also sets the name fields of the paper's run names: P0, CplusO_name, CtoO_name, aN_suffix ('' without N).
    """
    c_plus_o, c_to_o, x_n = float(params["CplusO"]), float(params["CtoO"]), float(params.get("X_N", 0))
    a_h = (1 - x_n) * (1 - c_plus_o)
    a_c = c_to_o / (1 + c_to_o) * (1 - x_n - a_h)
    params.update(a_h=a_h, a_c=a_c, a_o=1 - x_n - a_h - a_c, a_n=x_n)

    params["P0"] = _format_e_num(float(params["surface_pressure"]))
    params["CplusO_name"] = _format_e_num(c_plus_o)
    params["CtoO_name"] = f"{c_to_o:.1f}" if c_to_o == int(c_to_o) else f"{c_to_o:.10g}"
    params["aN_suffix"] = f"_aN={_format_e_num(x_n)}" if x_n else ""


DERIVE_HOOKS = {
    "c_plus_o_abundances": c_plus_o_abundances,
    "cia_pair_mixing": cia_pair_mixing,
    "early_mars_mixing": early_mars_mixing,
}
