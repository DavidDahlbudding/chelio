"""
`runner: session` of run_grid.py: every run is ONE HELIOS run in a shared HeliosSession (opacities stay loaded),
with the composition from a pre-computed chemistry grid (chem_grid) instead of the HELIOS <-> GGchem coupling loop,
optionally warm-started from an earlier solution.

Settings (grid file `session:` block, defaults in SESSION_DEFAULTS; also `session:` in config.yaml):
  chem_dT, chem_t_min, chem_t_max, chem_p_spacing ("opacity", "layers" or dex), chem_eqcond: chemistry grid
  max_iter                  HELIOS maximum_number_of_iterations (radiation and convection loop each)
  rad_max_iter              cold start: cap of the radiation-only loop, then convection loop (default max_iter)
  cold_start_temperature    isothermal start [K] (HELIOS / run_coupled.py: 500 K)
  min_conv_iter             minimum convection-loop iterations (HELIOS: 400)
  stitch_flip_limit, stitch_flip_start, stitch_after_iter, stitch_surface: helios_session.StitchPolicy
  write_tp_every            intermediate T-P profiles every n iterations (0: none)
  tp_coupling_every         write <name>_tp_coupling_<i>.dat every n iterations of the radiation and of the
                            convection loop (default 100, 0: only the final profile); i = radiation + convection
                            iterations so far, the final profile has the highest i
  warm_start                none | previous | nearest | <parameter> (next-higher value of that parameter, as for the
                            coupled runner) - from runs of this grid that have a <name>_restart.npz
  init_from                 directory with runs of the same names to start from (e.g. an earlier session grid)
Column model (chelio_sim/column.py; used as soon as one of these differs from its default; each key may also be
set per run in `grid` / `cases`, dicts are then merged into the session block's dict):
  chemistry_source          grid (chemistry grid as above) | constant_exact (constant_mixing_ratios, no grid) |
                            callable (vmr_callable)
  vmr_callable, vmr_callable_kwargs   "package.module:name" -> f(p [dyn/cm^2], T [K], species) -> {species: VMR}
                            (with kwargs: name(**kwargs) returns f)
  condensibles              (with chemistry: equilibrium, i.e. a GGchem grid, not recommended for now: see
                            docs/column_model_handoff.md, 'Known problem: GGchem grids and condensibles')
                            [] | auto (all active species with saturation data, thermo.AUTO_CONDENSIBLES) |
                            list, e.g. [H2O, NH3]: cold trap, moist adiabat, inhibition (relative_humidity -> H2O)
  convection                dict, see convection.ConvectionSettings: criterion (schwarzschild | ledoux), moist,
                            inhibition (experimental), inhibition_mode (jump | radiative), jump_width, semiconvection
                            (onset_only | strict), multi (sum | dominant),
                            thermo_source (host | table), inhibition_hysteresis, moist_tolerance, unfreeze,
                            max_adjust_iter, parcel_substeps
  surface_pressure_model    dict, see surface.py: callable, kwargs, update_every, relax, tol, p_min, p_max,
                            max_factor, settle (pressures in dyn/cm^2); the run parameter surface_pressure is
                            the starting value
  composition_damping       0 <= w < 1: log-VMR relaxation of the composition handed to HELIOS
  vapour_supply             running_max (default: a decrease of a condensible's base VMR with height is taken as
                            condensation and redone by the cold trap) | saturated | base, see column.ColumnModel
Per-run outputs (<out_dir>/<name>/): HELIOS output, <name>_tp_coupling_<i>.dat (T-P, bar/K, full precision,
every tp_coupling_every iterations; final = highest i),
<name>_restart.npz, <name>_session.json (convergence, iterations, timings, stitching, grid clamping, column model),
vertical_mix_0.dat (VMRs along the final profile, mixfile format), species.dat, delad_chelio.dat, run.log,
<name>_column.dat (column model: per-layer thermodynamics, thresholds, saturation, inhibition).
"""
import contextlib
import copy
import logging
import os
import shutil
import sys
import time

import numpy as np
import yaml

from chelio_sim import abundances, chem_grid, mixfile_utils, run_modes, species as species_utils
from chelio_sim import column, composition, convection, surface, thermo

log = logging.getLogger(__name__)

CHELIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SESSION_DEFAULTS = {
    "chem_dT": 50.0,
    "chem_t_min": 50.0,
    "chem_t_max": 2000.0,
    "chem_p_spacing": "opacity",
    "chem_eqcond": True,
    "max_iter": 30000,
    "rad_max_iter": None,
    "cold_start_temperature": 500.0,
    "min_conv_iter": 400,
    "stitch_flip_limit": 100,
    "stitch_after_iter": 5000,
    "stitch_surface": False,
    "stitch_flip_start": 100,
    "write_tp_every": 0,
    "tp_coupling_every": 100,
    "warm_start": "none",
    "init_from": None,
    # column model
    "chemistry_source": "grid",
    "vmr_callable": None,
    "vmr_callable_kwargs": None,
    "condensibles": [],
    "convection": None,
    "surface_pressure_model": None,
    "composition_damping": 0.0,
    "vapour_supply": "running_max",
}
# column-model settings that a run (grid / cases entry) may override; dicts are merged with the session block
COLUMN_KEYS = ("chemistry_source", "vmr_callable", "vmr_callable_kwargs", "condensibles", "convection",
               "surface_pressure_model", "composition_damping", "vapour_supply")
# placeholder kappa when the column model computes kappa and c_p itself (HELIOS needs a value at setup)
DUMMY_KAPPA = 0.2857
# run parameters of the session runner (passed by run_grid.py like CLI flags of the other runners)
SESSION_FLAGS = {*run_modes.SIM_PARAMS, "chemistry", "outgas_or_manual", "active_species", "constant_mixing_ratios", "cia_pairs",
                 "relative_humidity", "helios_param_file", *run_modes.CLOUD_PARAMS, run_modes.CLOUD_PATH_PARAM}


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path=None):
    with open(path or os.path.join(CHELIO_ROOT, "config.yaml")) as f:
        return _expand_env(yaml.safe_load(f))


def session_settings(config, spec_session=None):
    return {**SESSION_DEFAULTS, **(config.get("session") or {}), **(spec_session or {})}


def task_settings(settings, params):
    """Session settings of one run: column-model keys among the run parameters override the session block
    (dict values such as convection: {inhibition: false} are merged into the block's dict)."""
    out = dict(settings)
    for key in COLUMN_KEYS:
        if key in params:
            value = params[key]
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                value = {**out[key], **value}
            out[key] = value
    return out


def uses_column_model(settings):
    return (settings.get("chemistry_source", "grid") != "grid" or bool(settings.get("condensibles"))
            or bool(settings.get("convection")) or bool(settings.get("surface_pressure_model"))
            or float(settings.get("composition_damping") or 0) > 0)


def build_surface_pressure(spec):
    if not spec:
        return surface.SurfacePressure()
    spec = dict(spec)
    func = composition.load_callable(spec.pop("callable"), spec.pop("kwargs", None))
    return surface.SurfacePressure(func, **spec)


def build_column_model(settings, sim_p, active, grid, relative_humidity):
    """ColumnModel from the session settings (see the module docstring)."""
    source = settings.get("chemistry_source", "grid")
    if source == "grid":
        comp = composition.GridComposition(grid)
    elif source == "constant_exact":
        if sim_p["chemistry"] != "constant":
            raise ValueError("chemistry_source constant_exact needs chemistry: constant (constant_mixing_ratios)")
        comp = composition.ConstantComposition(sim_p["constant_mixing_ratios"])
    elif source == "callable":
        if not settings.get("vmr_callable"):
            raise ValueError("chemistry_source callable needs vmr_callable: package.module:name")
        func = composition.load_callable(settings["vmr_callable"], settings.get("vmr_callable_kwargs"))
        comp = composition.CallableComposition(func, label=str(settings["vmr_callable"]))
    else:
        raise ValueError(f"chemistry_source must be grid, constant_exact or callable, got {source!r}")
    names = settings.get("condensibles") or []
    if names == "auto":
        names = [s for s in thermo.AUTO_CONDENSIBLES if s in active]
    elif isinstance(names, str):
        names = [s.strip() for s in names.split(",") if s.strip()]
    missing = [s for s in names if s not in active]
    if missing:
        raise ValueError(f"condensibles {missing} are not active species ({active})")
    rh = 1.0 if relative_humidity is None else float(relative_humidity)
    conds = [thermo.Condensible(s, rh=rh if s == "H2O" else 1.0) for s in names]
    return column.ColumnModel(comp, conds, convection.ConvectionSettings.from_dict(settings.get("convection")),
                              build_surface_pressure(settings.get("surface_pressure_model")),
                              composition_damping=float(settings.get("composition_damping") or 0.0),
                              vapour_supply=settings.get("vapour_supply") or "running_max")


class _Tee:
    """File-like object writing to several streams (HELIOS' print output -> run.log [+ console])."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def _run_log(run_dir, level=logging.INFO, echo=False):
    """
    <run_dir>/run.log for one run, as for the coupled runner: chelio log messages (>= level) and everything HELIOS
    prints (stdout) go into it; with echo the printed output also still goes to the console.
    """
    f = open(os.path.join(run_dir, "run.log"), "a", buffering=1)
    handler = logging.StreamHandler(f)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(min(level, old_level) if old_level else level)
    stdout = _Tee(f, sys.stdout) if echo else f
    try:
        with contextlib.redirect_stdout(stdout):
            yield
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        f.close()


def run_task(session, name, out_dir, params, config, settings, init=None, echo=False):
    """
    One run (name, run parameters like the CLI flags of run_coupled.py). Returns the session result dict
    (raises on errors). Output in <out_dir>/<name>/; log and HELIOS' printed output in <name>/run.log
    (echo: printed output also to the console).
    """
    out_root = os.path.join(CHELIO_ROOT, out_dir)
    run_dir = os.path.join(out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    run_log = contextlib.ExitStack()
    level = getattr(logging, str((config.get("logging") or {}).get("level", "INFO")).upper(), logging.INFO)
    run_log.enter_context(_run_log(run_dir, level=level, echo=echo))
    try:
        config = copy.deepcopy(config)
        sim_p = config["simulation_params"]
        for key, value in params.items():
            if key == "constant_mixing_ratios" and isinstance(value, str):
                value = run_modes.parse_mixing_ratios(value)
            if key in SESSION_FLAGS and key not in ("relative_humidity", "helios_param_file", "cia_pairs"):
                sim_p[key] = value

        class _Args:
            pass
        args = _Args()
        args.relative_humidity = params.get("relative_humidity")
        args.constant_mixing_ratios = params.get("constant_mixing_ratios")
        for warning in run_modes.resolve_run_modes(config, args):
            log.warning(f"WARNING: {warning}")
        if sim_p["outgas_or_manual"] == "outgas" and sim_p["chemistry"] == "equilibrium":
            raise NotImplementedError("runner: session supports manual abundances and constant chemistry (no atmodeller yet)")
        chemistry = sim_p["chemistry"]
        log.info(f"--- Session run {name}: {chemistry} chemistry from a pre-computed grid ---")

        p_toa, p_boa = float(sim_p["toa_pressure"]), float(sim_p["surface_pressure"])
        use_column = uses_column_model(settings)
        surface_spec = settings.get("surface_pressure_model") or {}
        p_boa_grid = max(p_boa, float(surface_spec.get("p_max", p_boa)))
        if surface_spec and init and str(init).endswith(".npz"):
            # variable surface pressure: start on the grid of the earlier solution (exact warm start)
            with np.load(init) as d:
                p_boa = float(d["p_int"][0])
            log.info(f"Variable surface pressure: starting at the earlier run's {p_boa * 1e-6:.6g} bar")
            p_boa_grid = max(p_boa_grid, p_boa)
        active = species_utils.resolve_active_species(sim_p)
        species_file = os.path.join(run_dir, "species.dat")
        written = species_utils.write_species_file(species_file, active,
                                                   vmrs={s: species_utils.NEGLIGIBLE_VMR for s in active},
                                                   cia_pairs=species_utils.parse_cia_pairs(params.get("cia_pairs")))
        log.info(f"Species file: {', '.join(written)}")

        grid_opts = {"dT": settings["chem_dT"], "t_min": settings["chem_t_min"], "t_max": settings["chem_t_max"],
                     "p_spacing": settings["chem_p_spacing"], "eqcond": settings["chem_eqcond"]}
        cache_dir = os.path.join(out_root, "chem_grids")
        conv_settings = convection.ConvectionSettings.from_dict(settings.get("convection"))
        need_grid = not use_column or settings.get("chemistry_source", "grid") == "grid" \
            or conv_settings.thermo_source == "table"
        grid = None
        if not need_grid:
            pass
        elif chemistry == "equilibrium":
            abundances.calculate_abundances_manual(output_dir="ggchem", a_H=sim_p["a_h"], a_C=sim_p["a_c"],
                                                   a_O=sim_p["a_o"], a_N=sim_p["a_n"])
            abund_file = os.path.join(CHELIO_ROOT, "ggchem_inputs", "abundances.in")
            shutil.copy(abund_file, run_dir)
            grid = chem_grid.get_or_build(cache_dir, "equilibrium", active, p_toa, p_boa_grid, grid_opts,
                                          abundances_file=abund_file)
        else:
            rh = params.get("relative_humidity")
            grid = chem_grid.get_or_build(cache_dir, "constant", active, p_toa, p_boa_grid, grid_opts,
                                          mixing_ratios=sim_p["constant_mixing_ratios"],
                                          relative_humidity=1.0 if rh is None else float(rh))
        delad_path = os.path.join(run_dir, "delad_chelio.dat")
        if grid is not None:
            grid.write_delad_table(delad_path)
        column_model = None
        if use_column:
            column_model = build_column_model(settings, sim_p, active, grid, params.get("relative_humidity"))
            log.info(f"Column model: {column_model!r}")
        kappa = {"kappa_value": "file", "kappa_file_path": delad_path}
        if use_column and conv_settings.thermo_source == "host":
            kappa = {"kappa_value": DUMMY_KAPPA}  # kappa and c_p come from the column model

        cloud_flags = run_modes.resolve_clouds(sim_p, p_boa=p_boa, p_toa=p_toa, chelio_path=CHELIO_ROOT)
        param_file = os.path.join(CHELIO_ROOT, params.get("helios_param_file") or os.path.join("helios_inputs", "param.dat"))
        helios_params = {
            "parameter_file": param_file,
            "name": name,
            "output_directory": out_root.rstrip("/") + "/",
            "toa_pressure": p_toa,
            "boa_pressure": p_boa,
            "internal_temperature": sim_p["internal_temp"],
            "surface_albedo": sim_p["surface_albedo"],
            "path_to_species_file": species_file,
            **kappa,
            "maximum_number_of_iterations": int(settings["max_iter"]),
            "radiative_equilibrium_criterion": config["coupling"]["rad_eq_criterion"],
            **cloud_flags,
        }
        if init:
            log.info(f"Warm start from {init}")
        result = session.run(
            helios_params, vmr_provider=None if use_column else grid.provider(), column_model=column_model, init=init,
            cold_start_temperature=settings["cold_start_temperature"], min_conv_iter=settings["min_conv_iter"],
            rad_max_iter=settings["rad_max_iter"], write_tp_every=settings["write_tp_every"],
            tp_coupling_every=settings["tp_coupling_every"],
            stitch_flip_limit=settings["stitch_flip_limit"], stitch_after_iter=settings["stitch_after_iter"],
            stitch_surface=settings["stitch_surface"], stitch_flip_start=settings["stitch_flip_start"],
        )
        if column_model is not None:
            write_vertical_mix(os.path.join(run_dir, "vertical_mix_0.dat"), column_model.species,
                               lambda p_bar, T: column_model.vmr_profile(np.asarray(p_bar) * 1e6, T),
                               result["T_profile"])
        else:
            write_vertical_mix(os.path.join(run_dir, "vertical_mix_0.dat"), grid.species, grid.vmr_profile,
                               result["T_profile"])
        stats = result.get("vmr_provider_stats") or {}
        if stats.get("frac_T_clamped", 0) > 0:
            log.warning(f"T profile left the chemistry grid ({stats['T_seen']} K vs grid {stats['T_grid']} K) in "
                        f"{100 * stats['frac_T_clamped']:.2f}% of the evaluations: VMRs clamped at the edge")
        log.info(f"{name}: converged={result['converged']}, rad_iter={result['rad_iter']}, conv_iter={result['conv_iter']}, "
                 f"{result['t_total_s']:.0f} s (setup {result['t_setup_s']:.0f} s)")
        return result
    except Exception:
        log.exception(f"Session run {name} failed")
        raise
    finally:
        run_log.close()


def write_vertical_mix(path, species, vmr_profile, profile):
    """VMRs (vmr_profile(p_bar, T) -> (n, n_species)) along the final T-P profile (BOA first), in the mixfile format
    of convert_ggchem_to_helios."""
    p = np.asarray(profile["p_bar"])
    T = np.asarray(profile["T"])
    vmr = vmr_profile(p, T)
    weights = np.array([chem_grid.species_lib[s].weight for s in species])
    mu = vmr @ weights / vmr.sum(axis=1)
    n_tot = p * 1e6 / (mixfile_utils.kB * T)
    header = ["P(bar)", "T(k)", "n_<tot>(cm-3)", "m(u)", "e-", *species]
    data = np.column_stack([p, T, n_tot, mu, np.zeros_like(p), vmr])
    header_string = "\t".join(h + " " * max(16 - len(h), 0) for h in header)
    np.savetxt(path, data, header=header_string, fmt="%.10e", comments="", delimiter="\t")


def is_done(out_dir, name):
    path = os.path.join(CHELIO_ROOT, out_dir, name, f"{name}_coupling_convergence.dat")
    restart = os.path.join(CHELIO_ROOT, out_dir, name, f"{name}_restart.npz")
    return os.path.exists(restart) and os.path.exists(path) and open(path).read().strip() == "1"


def restart_file(out_dir, name):
    path = os.path.join(CHELIO_ROOT, out_dir, name, f"{name}_restart.npz")
    return path if os.path.exists(path) else None


def choose_warm_start(mode, task, tasks, out_dir, completed):
    """
    Restart file to start `task` from, or None.
      previous  last converged run of this invocation
      nearest   run of this grid (with a restart file) closest in the numeric run parameters that differ
                (log10 for positive parameters spanning more than a decade, each scaled by its range)
      <param>   run with the next-higher value of <param>, all other run parameters equal
    """
    if mode in (None, "none", False):
        return None
    if mode == "previous":
        for name in reversed(completed):
            path = restart_file(out_dir, name)
            if path:
                return path
        return None

    candidates = [t for t in tasks if t.name != task.name and restart_file(out_dir, t.name)]
    if not candidates:
        return None
    if mode == "nearest":
        keys = sorted(k for k in task.params if k in SESSION_FLAGS and isinstance(task.params[k], (int, float))
                      and not isinstance(task.params[k], bool))
        scale = {}
        for k in keys:
            values = np.array([float(t.params[k]) for t in tasks if k in t.params])
            log_axis = np.all(values > 0) and values.max() / values.min() > 10
            v = np.log10(values) if log_axis else values
            scale[k] = (log_axis, max(v.max() - v.min(), 1e-30))

        def dist(t):
            d = 0.0
            for k in keys:
                if k not in t.params:
                    return np.inf
                a, b = float(task.params[k]), float(t.params[k])
                log_axis, s = scale[k]
                d += ((np.log10(a) - np.log10(b)) if log_axis else (a - b)) ** 2 / s ** 2
            return d
        order = {name: i for i, name in enumerate(completed)}
        best = min(candidates, key=lambda t: (dist(t), -order.get(t.name, -1)))
        return restart_file(out_dir, best.name)

    axis = mode
    value = float(task.params[axis])
    same = [t for t in candidates if float(t.params.get(axis, -np.inf)) > value
            and all(t.params.get(k) == v for k, v in task.params.items() if k != axis)]
    if not same:
        return None
    return restart_file(out_dir, min(same, key=lambda t: float(t.params[axis])).name)


class SessionGridRunner:
    """Runs the tasks of a `runner: session` grid one after another in one HeliosSession (created on first use)."""

    def __init__(self, spec, tasks, report=print, verbose=False):
        self.spec = spec
        self.verbose = verbose  # HELIOS' printed output also to the console (always in the run's run.log)
        self.tasks = tasks
        self.report = report
        self.config = load_config(os.path.join(CHELIO_ROOT, spec["config"]) if spec.get("config") else None)
        self.settings = session_settings(self.config, spec.get("session"))
        if spec.get("warm_start") and "warm_start" not in (spec.get("session") or {}):
            self.settings["warm_start"] = spec["warm_start"]
        self.out_dir = spec["out_dir"]
        self.completed = []
        self.session = None

    def run(self, task):
        """Returns (status, result or exception or None); status: done, skipped, failed."""
        if is_done(self.out_dir, task.name):
            self.completed.append(task.name)
            return "skipped", None
        if self.session is None:
            from chelio_sim.helios_session import HeliosSession
            t0 = time.time()
            self.session = HeliosSession()
            self.report(f"HELIOS session started in {time.time() - t0:.1f} s")
        if self.settings.get("init_from"):
            init = restart_file(self.settings["init_from"], task.name)
            if init is None:
                self.report(f"init_from: no restart file for {task.name} in {self.settings['init_from']}, cold start")
        else:
            init = choose_warm_start(self.settings["warm_start"], task, self.tasks, self.out_dir, self.completed)
        params = dict(task.params)
        if self.spec.get("helios_param") and "helios_param_file" not in params:
            params["helios_param_file"] = self.spec["helios_param"]
        try:
            result = run_task(self.session, task.name, self.out_dir, params, self.config,
                              task_settings(self.settings, params), init=init, echo=self.verbose)
        except Exception as e:  # traceback in the run's run.log
            return "failed", e
        self.completed.append(task.name)
        return "done", result
