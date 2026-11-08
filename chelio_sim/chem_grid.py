"""
Pre-computed chemistry on a (T, P) grid, for HELIOS runs whose composition follows the temperature during the run
(helios_session.HeliosSession with `vmr_provider=grid.provider()`), instead of the HELIOS <-> GGchem coupling loop.

Two kinds of grids:
  equilibrium  GGchem at every grid point: model_dim 1, model_struc 5 (a list of independent P-T points), with
               equilibrium condensation (model_eqcond, optional) but remove_condensates .false.: a grid point
               knows nothing about the layers below it, so rain-out / cold trapping (remove_condensates .true.,
               used by the coupled runs) cannot be tabulated. With rain-out the composition depends on the
               whole profile (see the note in README "Pre-computed chemistry").
  constant     constant mixing ratios capped by saturation (mixfile_utils.cap_vmrs_by_saturation), the
               "pure condensation" chemistry of run_coupled.py --chemistry constant. Local, so the grid is exact
               up to interpolation.

Grid axes (both kinds):
  T: linear, t_min..t_max in steps of dT (default 50 K = the opacity grid of r50_kdistr)
  P: from 1e-6 bar (top of the opacity grid; above it the composition is clamped like the opacities) to p_boa
     "opacity"   10^(k/3) bar*1e-6 (the opacity grid: 1/3 dex)
     "layers"    HELIOS' layer and interface pressures for p_toa, p_boa (and nlayer, default HELIOS' automatic):
                 exact in P, only T is interpolated; one grid per surface pressure
     <number>    constant spacing in dex, aligned with 1 dyn/cm^2 (e.g. 1/6)
The VMRs are interpolated bilinearly in (T, log10 P) in log10(VMR), clamped at the grid edges (counted in
`provider.stats()`). The kappa/delad table for HELIOS is computed from the grid composition at every (T, P)
(write_delad_table). export_fastchem() writes the grid in HELIOS' FastChem format (chem.dat) for stock HELIOS
(species file mixing ratio "FastChem"), which resamples it onto the opacity grid and interpolates linearly.

Grids are cached in <cache_dir>/<kind>_<hash>.npz (hash over composition, GGchem settings, axes and species),
so a T_int sweep at fixed composition computes the grid once.
"""
import hashlib
import json
import logging
import os
import shutil
import subprocess
from math import ceil

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from chelio_sim import mixfile_utils
from chelio_sim.mixfile_utils import species_lib

log = logging.getLogger(__name__)

CHELIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GGCHEM_PARAM_TEMPLATE = os.path.join(CHELIO_ROOT, "ggchem_inputs", "param.in")
OPACITY_DLOGP = 1.0 / 3.0       # r50_kdistr pressure grid: 10^(k/3) dyn/cm^2
P_MIN_BAR = 1e-6                # lowest pressure of the chemistry grids (= lowest pressure of r50_kdistr)
LOG_VMR_FLOOR = -60.0           # log10(VMR) floor for interpolation
GRID_DEFAULTS = {"dT": 50.0, "t_min": 50.0, "t_max": 2000.0, "p_spacing": "opacity", "eqcond": True}
# files in $GGCHEM_PATH used for grids (separate from the coupled runs' abund_helios.in / pt_helios.in)
GGCHEM_ABUND_FILE = "abund_chemgrid.in"
GGCHEM_STRUC_FILE = "pt_chemgrid.in"
GGCHEM_PARAM_FILE = os.path.join("input", "param_chemgrid.in")


# --- axes ---

def temperature_axis(t_min, t_max, dT):
    n = int(round((t_max - t_min) / dT))
    return t_min + dT * np.arange(n + 1)


def helios_pressure_levels(p_toa, p_boa, nlayer=None):
    """HELIOS' layer + interface pressures [dyn/cm^2] (host_functions.calculate_pressure_levels), ascending."""
    if nlayer is None:
        nlayer = int(np.ceil(10.5 * np.log10(p_boa / p_toa)))
    levels = [p_boa * (p_toa / p_boa) ** (i / (2 * nlayer - 1)) for i in range(2 * nlayer)]
    levels.append(p_toa * (p_toa / p_boa) ** (1 / (2 * nlayer - 1)))
    return np.sort(np.array(levels))


def pressure_axis(p_toa, p_boa, spacing="opacity", nlayer=None):
    """
    Pressure axis [bar] from max(p_toa, P_MIN_BAR) to p_boa (given in dyn/cm^2). Like the opacities, the
    composition above P_MIN_BAR (the top of the opacity grid) is that of P_MIN_BAR (clamped).
    """
    if spacing == "layers":
        levels = helios_pressure_levels(p_toa, p_boa, nlayer) * 1e-6
        return np.concatenate(([P_MIN_BAR], levels[levels > P_MIN_BAR])) if levels[0] < P_MIN_BAR else levels
    dlogp = OPACITY_DLOGP if spacing == "opacity" else float(spacing)
    k_lo = max(int(np.floor(np.log10(p_toa) / dlogp + 1e-9)), int(round(np.log10(P_MIN_BAR * 1e6) / dlogp)))
    k_hi = int(np.ceil(np.log10(p_boa) / dlogp - 1e-9))
    return 10.0 ** (dlogp * np.arange(k_lo, k_hi + 1)) * 1e-6


# --- the grid ---

class GridVMRProvider:
    """vmr_provider for HeliosSession: log-VMR bilinear interpolation in (T, log10 P), clamped at the edges."""

    def __init__(self, grid):
        self.grid = grid
        logv = np.log10(np.clip(grid.vmr, 10.0 ** LOG_VMR_FLOOR, None))
        self._interp = RegularGridInterpolator((grid.T, np.log10(grid.P)), logv, method="linear")
        self._index = {s: i for i, s in enumerate(grid.species)}
        self.n_points = 0
        self.n_clamped_T = 0
        self.n_clamped_P = 0
        self.T_seen = [np.inf, -np.inf]

    def __call__(self, p_bar, T, species):
        T = np.asarray(T, float)
        logp = np.log10(np.asarray(p_bar, float))
        g = self.grid
        self.n_points += T.size
        self.n_clamped_T += int(np.sum((T < g.T[0]) | (T > g.T[-1])))
        self.n_clamped_P += int(np.sum((logp < np.log10(g.P[0]) - 1e-9) | (logp > np.log10(g.P[-1]) + 1e-9)))
        self.T_seen = [min(self.T_seen[0], float(T.min())), max(self.T_seen[1], float(T.max()))]
        pts = np.column_stack([np.clip(T, g.T[0], g.T[-1]), np.clip(logp, np.log10(g.P[0]), np.log10(g.P[-1]))])
        vmr = 10.0 ** self._interp(pts)
        return {s: vmr[:, self._index[s]] for s in species if s in self._index}

    def stats(self):
        n = max(self.n_points, 1)
        return {"grid": self.grid.label, "frac_T_clamped": self.n_clamped_T / n, "frac_P_clamped": self.n_clamped_P / n,
                "T_seen": self.T_seen, "T_grid": [float(self.grid.T[0]), float(self.grid.T[-1])]}

    def __repr__(self):
        return f"GridVMRProvider({self.grid.label})"


class ChemGrid:
    """VMRs on a (T, P) grid: T [K] (nT,), P [bar] (nP,), vmr (nT, nP, nS) for `species`."""

    def __init__(self, T, P, species, vmr, meta=None):
        self.T = np.asarray(T, float)
        self.P = np.asarray(P, float)
        self.species = list(species)
        self.vmr = np.asarray(vmr, float)
        self.meta = dict(meta or {})
        assert self.vmr.shape == (len(self.T), len(self.P), len(self.species))

    @property
    def label(self):
        return os.path.basename(self.meta.get("path", "")) or self.meta.get("kind", "grid")

    def save(self, path):
        self.meta["path"] = path
        np.savez(path, T=self.T, P=self.P, species=np.array(self.species), vmr=self.vmr, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            meta = json.loads(str(d["meta"]))
            meta["path"] = path
            return cls(d["T"], d["P"], [str(s) for s in d["species"]], d["vmr"], meta)

    def provider(self):
        return GridVMRProvider(self)

    def vmr_profile(self, p_bar, T):
        """VMRs (n, nS) along a T-P profile (same interpolation as the provider)."""
        out = GridVMRProvider(self)(p_bar, T, self.species)
        return np.column_stack([out[s] for s in self.species])

    def write_delad_table(self, out_path, t_step_max=10.0, dlogp_max=0.1):
        """
        HELIOS kappa/delad + c_p table (format of mixfile_utils.write_helios_delad_table) from the grid composition
        at every (T, P), on a uniform T (<= t_step_max) x uniform log10 P (<= dlogp_max dex) grid, as HELIOS'
        kappa interpolation needs.
        """
        t_axis = np.linspace(self.T[0], self.T[-1], int(ceil((self.T[-1] - self.T[0]) / t_step_max)) + 1)
        logp = np.log10(self.P)
        p_axis = 10.0 ** np.linspace(logp[0], logp[-1], int(ceil((logp[-1] - logp[0]) / dlogp_max)) + 1)

        kept, cps = [], []
        for j, s in enumerate(self.species):
            try:
                cps.append(mixfile_utils._get_janaf_cp_interpolator(s))
                kept.append(j)
            except Exception as e:
                if self.vmr[..., j].max() > 1e-20:
                    raise RuntimeError(f"No JANAF c_p for '{s}' (needed for the kappa table): {e}") from e
        provider = GridVMRProvider(self)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("This file contains the adiabatic coefficient (kappa, delad), heat capacity and entropy per unit mass\n")
            f.write("temp.[K]  press.[1e-6bar]  delad           c_p[erg mol^-1 K^-1]    log_S[log10(erg g^-1 K^-1)]\n")
            for T in t_axis:
                vmr = provider(p_axis, np.full_like(p_axis, T), self.species)
                v = np.column_stack([vmr[self.species[j]] for j in kept])
                v /= v.sum(axis=1, keepdims=True)
                cp = v @ np.array([c(T) for c in cps]) * mixfile_utils._J_TO_ERG
                delad = mixfile_utils.R / cp
                for P, d, c in zip(p_axis * 1e6, delad, cp):
                    f.write(f"{T:.6f}  {P:.6e}  {d:.6e}  {c:.6e}  0\n")
        log.info(f"Wrote kappa/delad table from the chemistry grid to {out_path}")
        return out_path

    def export_fastchem(self, directory):
        """chem.dat in HELIOS' FastChem format (Pbar, Tk, columns by FastChem name) for stock HELIOS."""
        os.makedirs(directory, exist_ok=True)
        names = [species_lib[s].fc_name for s in self.species]
        path = os.path.join(directory, "chem.dat")
        with open(path, "w") as f:
            f.write("Pbar Tk " + " ".join(names) + "\n")
            for iT, T in enumerate(self.T):
                for iP, P in enumerate(self.P):
                    f.write(f"{P:.8e} {T:.6f} " + " ".join(f"{v:.6e}" for v in self.vmr[iT, iP]) + "\n")
        return path


# --- builders ---

def build_constant_grid(mixing_ratios, T, P, relative_humidity=1.0):
    species = list(mixing_ratios)
    TT, PP = np.meshgrid(T, P, indexing="ij")
    vmr = mixfile_utils.cap_vmrs_by_saturation(PP.ravel(), TT.ravel(), species, [float(mixing_ratios[s]) for s in species],
                                               relative_humidity=relative_humidity)
    return ChemGrid(T, P, species, vmr.reshape(len(T), len(P), len(species)),
                    {"kind": "constant", "mixing_ratios": {s: float(v) for s, v in mixing_ratios.items()},
                     "relative_humidity": relative_humidity})


def _ggchem_param_file(template_path, eqcond):
    """GGchem parameters for independent points: model_dim 1, model_struc 5, no rain-out."""
    lines = open(template_path).read().splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if "! abund_pick" in line:
            out += [line, GGCHEM_ABUND_FILE]
            i += 2
            continue
        if "! model_eqcond" in line:
            line = f"{'.true.' if eqcond else '.false.':<22}! model_eqcond"
        elif "! remove_condensates" in line:
            line = f"{'.false.':<22}! remove_condensates"
        elif "! model_dim" in line:
            line = f"{'1':<22}! model_dim  (0,1,2)"
        elif "! model_struc" in line:
            out += [f"{'5':<22}! model_struc", GGCHEM_STRUC_FILE]
            i += 2
            continue
        elif "! verbose" in line:
            line = f"{'0':<22}! verbose"
        out.append(line)
        i += 1
    if not any("! remove_condensates" in l for l in out) and eqcond:
        out.insert(next(k for k, l in enumerate(out) if "! model_eqcond" in l) + 1, f"{'.false.':<22}! remove_condensates")
    return "\n".join(out) + "\n"


def _read_static_conc(path, species):
    """GGchem Static_Conc.dat -> (T, p_bar, VMRs (n, nS) of `species`); VMR = n_i / n_gas as convert_ggchem_to_helios."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty(0), np.empty(0), np.empty((0, len(species)))
    header = np.loadtxt(path, skiprows=2, max_rows=1, dtype=str)
    dims = np.genfromtxt(path, dtype=int, max_rows=1, skip_header=1)
    data = np.loadtxt(path, skiprows=3, ndmin=2)
    if data.size == 0:
        return np.empty(0), np.empty(0), np.empty((0, len(species)))
    n_gas = dims[0] + dims[1]
    logn = data[:, 3:4 + n_gas]
    n_tot = np.sum(10.0 ** logn, axis=1)
    cols = list(header[3:4 + n_gas])
    missing = [s for s in species if s not in cols]
    if missing:
        raise ValueError(f"Species {missing} not in GGchem output {path}")
    vmr = np.column_stack([10.0 ** logn[:, cols.index(s)] / n_tot for s in species])
    return data[:, 0], data[:, 2] * 1e-6, vmr


def build_ggchem_grid(abundances_file, T, P, species, eqcond=True, ggchem_path=None, template=GGCHEM_PARAM_TEMPLATE,
                      max_restarts=200):
    """
    GGchem on every (T, P) point. Points are run isobar by isobar (high to low P), T in snake order (hot->cold, then
    cold->hot) so that consecutive points are close. If GGchem stops at a point, the rest is run again without it;
    points without a result get the composition of the nearest successful point of the same isobar, capped by
    saturation (as mixfile_utils.append_profiles does for the coupled runs; H2 is not capped).
    """
    ggchem_path = ggchem_path or os.environ["GGCHEM_PATH"]
    shutil.copy(abundances_file, os.path.join(ggchem_path, GGCHEM_ABUND_FILE))
    with open(os.path.join(ggchem_path, GGCHEM_PARAM_FILE), "w") as f:
        f.write(_ggchem_param_file(template, eqcond))

    points = []
    for iP in range(len(P) - 1, -1, -1):
        iTs = range(len(T) - 1, -1, -1) if (len(P) - 1 - iP) % 2 == 0 else range(len(T))
        points += [(iT, iP) for iT in iTs]

    vmr = np.full((len(T), len(P), len(species)), np.nan)
    todo, restarts, failed = list(points), 0, []
    out_path = os.path.join(ggchem_path, "Static_Conc.dat")
    while todo:
        with open(os.path.join(ggchem_path, "structures", GGCHEM_STRUC_FILE), "w") as f:
            f.write("# P [bar], T [K]\n")
            for iT, iP in todo:
                f.write(f"{P[iP]:.10e} {T[iT]:.6f}\n")
        if os.path.exists(out_path):
            os.remove(out_path)
        log.info(f"GGchem grid: {len(todo)} points ({len(points)} total)")
        subprocess.run(["./ggchem", GGCHEM_PARAM_FILE], cwd=ggchem_path, input="\n" * 2000, text=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Tg, pg, v = _read_static_conc(out_path, species)
        n_done = len(Tg)
        for k in range(n_done):
            iT, iP = todo[k]
            if abs(Tg[k] - T[iT]) > 1e-3 * T[iT] or abs(np.log10(pg[k] / P[iP])) > 1e-3:
                raise RuntimeError(f"GGchem grid output row {k} (T={Tg[k]}, p={pg[k]}) does not match the input point "
                                   f"(T={T[iT]}, p={P[iP]})")
            vmr[iT, iP] = v[k]
        if n_done == len(todo):
            break
        failed.append(todo[n_done])
        todo = todo[n_done + 1:]
        restarts += 1
        if restarts > max_restarts:
            failed += todo
            log.warning(f"GGchem grid: giving up after {max_restarts} restarts")
            break
    try:
        os.remove(os.path.join(ggchem_path, "database.dat"))
    except FileNotFoundError:
        pass

    if failed:
        log.warning(f"GGchem grid: {len(failed)} of {len(points)} points failed, filled from the nearest point of the "
                    f"same isobar with saturation caps: {[(float(T[a]), float(P[b])) for a, b in failed[:10]]}...")
        for iT, iP in failed:
            ok = np.flatnonzero(np.all(np.isfinite(vmr[:, iP]), axis=1))
            if ok.size == 0:
                raise RuntimeError(f"GGchem failed on the whole isobar P = {P[iP]:g} bar")
            src = ok[np.argmin(np.abs(ok - iT))]
            vmr[iT, iP] = mixfile_utils.cap_vmrs_by_saturation([P[iP]], [T[iT]], species, vmr[src, iP][None, :],
                                                               uncapped=("H2",))[0]
    return ChemGrid(T, P, species, vmr, {"kind": "equilibrium", "eqcond": bool(eqcond), "remove_condensates": False,
                                         "n_failed": len(failed), "abundances": open(abundances_file).read()})


def get_or_build(cache_dir, kind, species, p_toa, p_boa, options=None, mixing_ratios=None, relative_humidity=1.0,
                 abundances_file=None):
    """Cached grid for a composition. options: dT, t_min, t_max, p_spacing, nlayer, eqcond (GRID_DEFAULTS)."""
    opts = {**GRID_DEFAULTS, **(options or {})}
    T = temperature_axis(float(opts["t_min"]), float(opts["t_max"]), float(opts["dT"]))
    P = pressure_axis(float(p_toa), float(p_boa), opts["p_spacing"], opts.get("nlayer"))
    species = [s for s in species if not s.startswith("CIA")]

    key = {"kind": kind, "species": species, "T": T.tolist(), "P": [float(f"{p:.8e}") for p in P]}
    if kind == "equilibrium":
        key.update(abundances=open(abundances_file).read(), eqcond=bool(opts["eqcond"]),
                   ggchem_param=open(GGCHEM_PARAM_TEMPLATE).read())
    else:
        key.update(mixing_ratios={s: float(v) for s, v in mixing_ratios.items()}, relative_humidity=relative_humidity)
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{kind}_{digest}.npz")
    if os.path.exists(path):
        log.info(f"Chemistry grid from cache: {path}")
        return ChemGrid.load(path)

    log.info(f"Building {kind} chemistry grid: {len(T)} T ({T[0]:g}-{T[-1]:g} K) x {len(P)} P "
             f"({P[0]:.2e}-{P[-1]:.2e} bar), {len(species)} species")
    if kind == "equilibrium":
        grid = build_ggchem_grid(abundances_file, T, P, species, eqcond=opts["eqcond"])
    elif kind == "constant":
        species = [s for s in mixing_ratios]
        grid = build_constant_grid(mixing_ratios, T, P, relative_humidity)
    else:
        raise ValueError(f"unknown grid kind {kind!r}")
    grid.meta["options"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in opts.items()}
    grid.save(path)
    with open(path[:-4] + ".json", "w") as f:
        json.dump({k: v for k, v in grid.meta.items()}, f, indent=1)
    return grid
