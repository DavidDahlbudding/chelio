"""
Base compositions for the session runner's column model (chelio_sim.column).

A composition gives the "base" mole fractions at given (p, T) points: local chemistry before the column processes
(cold trap / rain-out of the condensibles), which the column model applies on top (column.ColumnModel). Contract,
in HELIOS' cgs units:

    composition(p, T, species) -> {species: VMR array}      p [dyn/cm^2], T [K], 1D arrays of equal length

(helios_session's older vmr_provider and chem_grid's GridVMRProvider take p in bar; GridComposition converts.)

Species that are not returned get the negligible VMR of the species file (1e-29). Optional methods: stats() (dict
for <name>_session.json), a stable __repr__ (part of the warm-start fingerprint).

  GridComposition      chem_grid.ChemGrid (log-VMR interpolation in (T, log P), equilibrium or constant chemistry).
                       Equilibrium grids include local condensation, and their interpolated VMRs of condensibles lie
                       far below saturation at low T: NOT RECOMMENDED together with condensibles / moist convection
                       for now (column.ColumnModel warns; docs/column_model_handoff.md, 'Known problem: GGchem grids
                       and condensibles')
  ConstantComposition  fixed mixing ratios (normalised, not capped: the column model caps the condensibles)
  CallableComposition  any function with the contract above (e.g. "mypackage.mymodule:my_vmrs" in a grid file);
                       bar_input=True wraps a function that takes p in bar (e.g. an old vmr_provider)
"""
import importlib
import logging

import numpy as np

log = logging.getLogger(__name__)


def load_callable(spec, kwargs=None):
    """'package.module:name' -> object; with kwargs the object is called as a factory: name(**kwargs)."""
    if callable(spec):
        obj = spec
    else:
        if ":" not in str(spec):
            raise ValueError(f"Callable must be given as 'package.module:name', got {spec!r}")
        module, name = str(spec).split(":", 1)
        obj = getattr(importlib.import_module(module), name)
    return obj(**kwargs) if kwargs else obj


class GridComposition:
    """Wraps a chem_grid.ChemGrid (its GridVMRProvider: interpolation, clamping statistics)."""

    def __init__(self, grid):
        self.grid = grid
        self.provider = grid.provider()

    def __call__(self, p, T, species):
        return self.provider(np.asarray(p, float) * 1e-6, T, species)  # the grid works in bar

    def stats(self):
        return self.provider.stats()

    def __repr__(self):
        return f"GridComposition({self.grid.label})"


class ConstantComposition:
    """Constant mixing ratios (normalised over the species asked for), without saturation caps."""

    def __init__(self, mixing_ratios):
        self.mixing_ratios = {s: float(v) for s, v in mixing_ratios.items()}
        if any(v < 0 for v in self.mixing_ratios.values()) or sum(self.mixing_ratios.values()) <= 0:
            raise ValueError(f"Invalid constant mixing ratios {mixing_ratios}")

    def __call__(self, p, T, species):
        n = len(np.atleast_1d(p))
        total = sum(self.mixing_ratios.get(s, 0.0) for s in species)
        return {s: np.full(n, self.mixing_ratios[s] / total) for s in species if s in self.mixing_ratios}

    def stats(self):
        return {"mixing_ratios": self.mixing_ratios}

    def __repr__(self):
        return "ConstantComposition(" + ",".join(f"{s}={v:g}" for s, v in self.mixing_ratios.items()) + ")"


class CallableComposition:
    """Any function f(p [dyn/cm^2], T [K], species) -> {species: VMR array} (bar_input: f takes p in bar)."""

    def __init__(self, func, label=None, bar_input=False):
        self.func = func
        self.label = label or getattr(func, "__qualname__", None) or repr(func)
        self.bar_input = bool(bar_input)

    def __call__(self, p, T, species):
        p = np.asarray(p, float) * (1e-6 if self.bar_input else 1.0)
        return self.func(p, np.asarray(T, float), species)

    def stats(self):
        return self.func.stats() if hasattr(self.func, "stats") else {}

    def __repr__(self):
        return f"CallableComposition({self.label})"
