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
  warm_start                none | previous | nearest | <parameter> (next-higher value of that parameter, as for the
                            coupled runner) - from runs of this grid that have a <name>_restart.npz
  init_from                 directory with runs of the same names to start from (e.g. an earlier session grid)
Per-run outputs (<out_dir>/<name>/): HELIOS output, <name>_tp_coupling_0.dat (final T-P, bar/K, full precision),
<name>_restart.npz, <name>_session.json (convergence, iterations, timings, stitching, grid clamping),
vertical_mix_0.dat (grid VMRs along the final profile, mixfile format), species.dat, delad_chelio.dat, run.log.
"""
import copy
import logging
import os
import shutil
import time

import numpy as np
import yaml

from chelio_sim import abundances, chem_grid, mixfile_utils, run_modes, species as species_utils

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
    "warm_start": "none",
    "init_from": None,
}
# run parameters of the session runner (passed by run_grid.py like CLI flags of the other runners)
SESSION_FLAGS = {*run_modes.SIM_PARAMS, "chemistry", "outgas_or_manual", "active_species", "constant_mixing_ratios",
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


def _file_logger(run_dir):
    handler = logging.FileHandler(os.path.join(run_dir, "run.log"), mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def run_task(session, name, out_dir, params, config, settings, init=None):
    """
    One run (name, run parameters like the CLI flags of run_coupled.py). Returns the session result dict
    (raises on errors). Output in <out_dir>/<name>/.
    """
    out_root = os.path.join(CHELIO_ROOT, out_dir)
    run_dir = os.path.join(out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    handler = _file_logger(run_dir)
    try:
        config = copy.deepcopy(config)
        sim_p = config["simulation_params"]
        for key, value in params.items():
            if key == "constant_mixing_ratios" and isinstance(value, str):
                value = run_modes.parse_mixing_ratios(value)
            if key in SESSION_FLAGS and key not in ("relative_humidity", "helios_param_file"):
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
        active = species_utils.resolve_active_species(sim_p)
        species_file = os.path.join(run_dir, "species.dat")
        species_utils.write_species_file(species_file, active, vmrs={s: species_utils.NEGLIGIBLE_VMR for s in active})

        grid_opts = {"dT": settings["chem_dT"], "t_min": settings["chem_t_min"], "t_max": settings["chem_t_max"],
                     "p_spacing": settings["chem_p_spacing"], "eqcond": settings["chem_eqcond"]}
        cache_dir = os.path.join(out_root, "chem_grids")
        if chemistry == "equilibrium":
            abundances.calculate_abundances_manual(output_dir="ggchem", a_H=sim_p["a_h"], a_C=sim_p["a_c"],
                                                   a_O=sim_p["a_o"], a_N=sim_p["a_n"])
            abund_file = os.path.join(CHELIO_ROOT, "ggchem_inputs", "abundances.in")
            shutil.copy(abund_file, run_dir)
            grid = chem_grid.get_or_build(cache_dir, "equilibrium", active, p_toa, p_boa, grid_opts, abundances_file=abund_file)
        else:
            rh = params.get("relative_humidity")
            grid = chem_grid.get_or_build(cache_dir, "constant", active, p_toa, p_boa, grid_opts,
                                          mixing_ratios=sim_p["constant_mixing_ratios"],
                                          relative_humidity=1.0 if rh is None else float(rh))
        delad_path = os.path.join(run_dir, "delad_chelio.dat")
        grid.write_delad_table(delad_path)

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
            "kappa_value": "file",
            "kappa_file_path": delad_path,
            "maximum_number_of_iterations": int(settings["max_iter"]),
            "radiative_equilibrium_criterion": config["coupling"]["rad_eq_criterion"],
            **cloud_flags,
        }
        if init:
            log.info(f"Warm start from {init}")
        result = session.run(
            helios_params, vmr_provider=grid.provider(), init=init,
            cold_start_temperature=settings["cold_start_temperature"], min_conv_iter=settings["min_conv_iter"],
            rad_max_iter=settings["rad_max_iter"], write_tp_every=settings["write_tp_every"],
            stitch_flip_limit=settings["stitch_flip_limit"], stitch_after_iter=settings["stitch_after_iter"],
            stitch_surface=settings["stitch_surface"], stitch_flip_start=settings["stitch_flip_start"],
        )
        write_vertical_mix(os.path.join(run_dir, "vertical_mix_0.dat"), grid, result["T_profile"])
        stats = result.get("vmr_provider_stats") or {}
        if stats.get("frac_T_clamped", 0) > 0:
            log.warning(f"T profile left the chemistry grid ({stats['T_seen']} K vs grid {stats['T_grid']} K) in "
                        f"{100 * stats['frac_T_clamped']:.2f}% of the evaluations: VMRs clamped at the edge")
        log.info(f"{name}: converged={result['converged']}, rad_iter={result['rad_iter']}, conv_iter={result['conv_iter']}, "
                 f"{result['t_total_s']:.0f} s (setup {result['t_setup_s']:.0f} s)")
        return result
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def write_vertical_mix(path, grid, profile):
    """Grid VMRs along the final T-P profile, in the mixfile format of convert_ggchem_to_helios."""
    p = np.asarray(profile["p_bar"])
    T = np.asarray(profile["T"])
    vmr = grid.vmr_profile(p, T)
    weights = np.array([chem_grid.species_lib[s].weight for s in grid.species])
    mu = vmr @ weights / vmr.sum(axis=1)
    n_tot = p * 1e6 / (mixfile_utils.kB * T)
    header = ["P(bar)", "T(k)", "n_<tot>(cm-3)", "m(u)", "e-", *grid.species]
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

    def __init__(self, spec, tasks, report=print):
        self.spec = spec
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
            result = run_task(self.session, task.name, self.out_dir, params, self.config, self.settings, init=init)
        except Exception as e:
            log.exception(f"Session run {task.name} failed")
            return "failed", e
        self.completed.append(task.name)
        return "done", result
