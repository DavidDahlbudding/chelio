"""
EXPERIMENTAL: static local grid refinement above the BOA for the session runner (option A of
claude-scratch/adaptive_grid_inhibited_layers.md). Two-pass use: run on HELIOS' uniform grid, then rerun (warm start)
with more layers just above the surface, where condensation-inhibited radiative layers form (Innes et al. 2023 used
200 layers with the bottom 100 in that region).

Layer density in x = log10(p_boa / p) (dex above the BOA), smooth step from the fine to the base density:

    rho(x) = rho_base + (rho_fine - rho_base) * sigma((depth - x) / ramp),   sigma = logistic

HELIOS' 2 nlayer pressure levels (interfaces at even, layer centres at odd indices, BOA = level 0, p_toa = top layer
centre) are placed at equal increments of the integral of rho; the TOA interface is extrapolated with the last
spacing (as HELIOS does). With rho_fine = rho_base this is exactly HELIOS' uniform grid. The region moves with the
BOA, so it also works with a variable surface pressure (regrid uses the same function).

Session setting (grid file `session:` block):

    grid_refinement:
      layers_per_dex: 200      # density inside the region
      base_layers_per_dex: 20  # density elsewhere (nlayer is computed from both and passed to HELIOS)
      depth_dex: auto          # region [BOA, BOA - depth] in dex; auto: from the warm-start run's <name>_column.dat
      ramp_dex: 0.03           # width of the smooth transition
      gamma_min: 0.5           # auto: saturated layers with Gamma >= gamma_min (or inhibited), contiguous from the BOA
      margin_dex: 0.05         # auto: added above the detected region
      min_depth_dex: 0.1       # auto: lower bound (also used when nothing is detected)
      pseudo_step_base: false  # true: pseudo-time step of HELIOS with the base grid's BOA layer thickness

HELIOS' interface-temperature kernel (temp_inter) assumes equal spacing; with a refined grid the session replaces it
by linear interpolation in ln p (identical on a uniform grid). HELIOS' radiative pseudo-time step uses the BOA layer
thickness for all layers (kernels.cu rad_temp_iter / conv_temp_iter), so a refined BOA layer scales every step up by
~rho_fine / rho_base. Option pseudo_step_base hands these kernels the base grid's BOA thickness instead (off by default).
2026-11-21 tests (1 bar, Innes+23 setup, radiative inhibition), T_s [K]:
    S = 500: 200/dex 316.4 (pseudo_step_base: 329.3), 400/dex 316.3
    S = 550: 200/dex 351.5, 400/dex did not converge (BOA swinging 276-342 K over 21000 iterations); 400/dex with
             inhibition_hysteresis 0.1: 352.8 (but hysteresis biases towards thicker inhibited layers); 400/dex with
             pseudo_step_base: 342.1
The converged profiles agree above ~0.8 bar and differ only in the top of the inhibited layer (0.80-0.88 bar), i.e.
there are several converged states (path dependence, as noted by Innes et al.): ~10-15 K uncertainty in T_s.
"""
import glob
import logging
import os
from dataclasses import dataclass, asdict

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GridRefinement:
    layers_per_dex: float = 200.0
    base_layers_per_dex: float = 20.0
    depth_dex: object = "auto"
    ramp_dex: float = 0.03
    gamma_min: float = 0.5
    margin_dex: float = 0.05
    min_depth_dex: float = 0.1
    pseudo_step_base: bool = False  # HELIOS' pseudo-time step with the base grid's BOA layer thickness (see helios_session)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown grid_refinement settings {sorted(unknown)}")
        g = cls(**d)
        if g.depth_dex != "auto" and not float(g.depth_dex) > 0:
            raise ValueError("grid_refinement.depth_dex must be 'auto' or > 0")
        if min(g.layers_per_dex, g.base_layers_per_dex, g.ramp_dex) <= 0:
            raise ValueError("grid_refinement: densities and ramp_dex must be > 0")
        return g

    def as_dict(self):
        return asdict(self)


def density(x, depth, rho_fine, rho_base, ramp):
    """Layers per dex at x dex above the BOA."""
    with np.errstate(over="ignore"):
        s = 1.0 / (1.0 + np.exp(-(depth - x) / ramp))
    return rho_base + (rho_fine - rho_base) * s


def _cumulative(D, depth, rho_fine, rho_base, ramp, m=20001):
    x = np.linspace(0.0, D, m)
    rho = density(x, depth, rho_fine, rho_base, ramp)
    R = np.concatenate(([0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))))
    return x, R


def n_layers(p_boa, p_toa, depth, rho_fine, rho_base, ramp):
    """Number of layers of the refined grid (integral of the density from the BOA to p_toa)."""
    D = np.log10(p_boa / p_toa)
    _, R = _cumulative(D, depth, rho_fine, rho_base, ramp)
    return int(np.ceil(R[-1]))


def pressure_levels(p_boa, p_toa, nlayer, depth, rho_fine, rho_base, ramp):
    """(p_lay, p_int) like host_functions.calculate_pressure_levels (p_int has nlayer + 1 entries, TOA extrapolated)."""
    D = np.log10(p_boa / p_toa)
    x, R = _cumulative(D, depth, rho_fine, rho_base, ramp)
    targets = np.linspace(0.0, R[-1], 2 * nlayer)          # 2 nlayer levels, BOA ... p_toa
    xl = np.interp(targets, R, x)
    levels = p_boa * 10.0 ** (-xl)
    p_lay = levels[1::2]
    p_int = levels[0::2]
    p_int = np.append(p_int, levels[-1] * levels[-1] / levels[-2])
    return p_lay, p_int


def auto_depth(column_file, g):
    """Depth [dex] of the region from an earlier run's <name>_column.dat (bottom -> top layers)."""
    lines = open(column_file).read().splitlines()
    names = lines[2].split()
    data = np.array([[float(v) for v in l.split()] for l in lines[3:]])
    c = {k: data[:, i] for i, k in enumerate(names)}
    p_boa = float(lines[1].split("p = ")[1].split()[0])
    sat = c.get("sat_H2O", np.zeros(len(data))) > 0
    hit = ((c["Gamma"] >= g.gamma_min) & sat) | (c["inhibited"] > 0)
    if not hit[0]:
        return float(g.min_depth_dex), None
    k = int(np.argmin(hit)) if not hit.all() else len(hit)
    p_top = c["p[dyn/cm2]"][k - 1]
    d = float(np.log10(p_boa / p_top)) + g.margin_dex
    return max(d, float(g.min_depth_dex)), float(p_top)


def resolve(spec, p_boa, p_toa, init=None):
    """Settings -> dict(depth, rho_fine, rho_base, ramp, nlayer, source) for one run, or None."""
    if not spec:
        return None
    g = GridRefinement.from_dict(spec)
    source = "fixed"
    p_top = None
    if g.depth_dex == "auto":
        files = glob.glob(os.path.join(os.path.dirname(init), "*_column.dat")) if init else []
        if files:
            depth, p_top = auto_depth(files[0], g)
            source = f"auto from {files[0]}"
        else:
            depth = float(g.min_depth_dex)
            source = "auto: no column file of the warm-start run, min_depth_dex"
    else:
        depth = float(g.depth_dex)
    out = {"depth_dex": depth, "rho_fine": float(g.layers_per_dex), "rho_base": float(g.base_layers_per_dex),
           "ramp_dex": float(g.ramp_dex), "pseudo_step_base": bool(g.pseudo_step_base), "source": source, "detected_p_top_dyn": p_top, "settings": g.as_dict()}
    out["nlayer"] = n_layers(p_boa, p_toa, depth, out["rho_fine"], out["rho_base"], out["ramp_dex"])
    return out


def levels_for(quant, ref):
    """Refined (p_lay, p_int) for HELIOS' current p_boa / p_toa / nlayer."""
    return pressure_levels(float(quant.p_boa), float(quant.p_toa), int(quant.nlayer), ref["depth_dex"],
                           ref["rho_fine"], ref["rho_base"], ref["ramp_dex"])


def interface_temperatures(T_lay, p_lay, p_int):
    """T at interfaces, linear in ln p between layer centres (extrapolated at BOA and TOA); equals HELIOS'
    temp_inter on a uniform grid."""
    lp, li = np.log(p_lay), np.log(p_int)
    T = np.empty(len(p_int))
    w = (li[1:-1] - lp[:-1]) / (lp[1:] - lp[:-1])
    T[1:-1] = T_lay[:-1] + w * (T_lay[1:] - T_lay[:-1])
    T[0] = T_lay[0] + (li[0] - lp[0]) / (lp[1] - lp[0]) * (T_lay[1] - T_lay[0])
    T[-1] = T_lay[-1] + (li[-1] - lp[-1]) / (lp[-2] - lp[-1]) * (T_lay[-2] - T_lay[-1])
    return T


def _check():
    import types
    p_boa, p_toa, n = 1e6, 1.0, 120
    # uniform limit reproduces HELIOS' grid
    p_lay, p_int = pressure_levels(p_boa, p_toa, n, 0.2, 20.0, 20.0, 0.03)
    lv = [p_boa * (p_toa / p_boa) ** (i / (2 * n - 1)) for i in range(2 * n)]
    ref_lay, ref_int = np.array(lv[1::2]), np.array(lv[0::2] + [p_toa * (p_toa / p_boa) ** (1 / (2 * n - 1))])
    assert np.allclose(p_lay, ref_lay, rtol=1e-6) and np.allclose(p_int, ref_int, rtol=1e-6), "uniform limit"
    # interface T equals HELIOS' temp_inter on the uniform grid
    T = 300 - 20 * np.log10(p_boa / ref_lay) + np.sin(np.arange(n))
    Ti = interface_temperatures(T, ref_lay, ref_int)
    Th = np.empty(n + 1)
    Th[1:-1] = T[:-1] + 0.5 * (T[1:] - T[:-1]); Th[0] = T[0] - 0.5 * (T[1] - T[0]); Th[-1] = T[-1] + 0.5 * (T[-1] - T[-2])
    assert np.allclose(Ti, Th, atol=1e-9), "temp_inter"
    # refined grid: counts, monotonic, smooth
    for d in (0.1, 0.3):
        nl = n_layers(p_boa, p_toa, d, 200.0, 20.0, 0.03)
        pl, pi = pressure_levels(p_boa, p_toa, nl, d, 200.0, 20.0, 0.03)
        lv = np.log10(np.sort(np.concatenate((pl, pi)))[::-1])
        dx = -np.diff(lv)
        ratio = np.max(np.maximum(dx[1:] / dx[:-1], dx[:-1] / dx[1:]))
        inside = np.sum(np.log10(p_boa / pl) < d)
        print(f"depth {d} dex: nlayer {nl}, layers in region {inside} ({inside / d:.0f}/dex), max neighbour ratio "
              f"{ratio:.3f}, top spacing {1 / (2 * dx[-2]):.1f} layers/dex")
        assert np.all(dx > 0)
    print("grid_refine check: ok")


if __name__ == "__main__":
    _check()
