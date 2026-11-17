"""
Variable surface pressure for the session runner's column model.

A user function maps the current column state to the surface pressure it implies:

    func(state: ColumnState) -> p_surf [dyn/cm^2]

During the convection loop (every `update_every` iterations, at an iteration where HELIOS' adaptive time step
stores its reference temperatures) the session moves p_surf towards that target, damped in log p:

    ln p_new = ln p + relax * (ln p_target - ln p),   |ln(p_new / p)| <= ln(max_factor),   p_min <= p_new <= p_max

and rebuilds the pressure grid with the same number of layers and the same TOA pressure (helios_session
SessionCompute.regrid). The run converges only when |ln(p_target / p)| < tol and at least `settle` iterations have
passed since the last change. Grid file (session block):

    surface_pressure_model:                          # the run parameter surface_pressure is the start value
      callable: chelio_sim.surface:fixed_inventory   # package.module:name
      kwargs: {species: N2, p_inventory: 1.0e6}     # optional: name(**kwargs) returns the function
      update_every: 200
      relax: 0.5
      tol: 1.0e-3
      p_max: 1.0e8                                   # dyn/cm^2; also the upper end of the chemistry grid

Example functions: fixed_inventory (column mass of one non-condensible species fixed, e.g. a fixed N2 inventory
with an ocean: the vapour adds to it), ocean_plus_inventory (p_s = p_inventory + RH p_sat(T_surf)).
"""
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ColumnState:
    """Snapshot of the column passed to the surface-pressure function. Pressures in dyn/cm^2, layers bottom->top."""
    p_lay: np.ndarray
    p_int: np.ndarray
    T_lay: np.ndarray
    T_int: np.ndarray
    T_surf: float
    species: list
    x_lay: np.ndarray          # (n, n_species) mole fractions
    x_int: np.ndarray          # (n+1, n_species)
    M: np.ndarray              # (n_species,) g/mol
    mu_lay: np.ndarray         # g/mol
    g: float                   # cm/s^2
    conv_layer: np.ndarray     # HELIOS flags (n+1, surface last)
    inhibited: np.ndarray      # (n,)
    condensibles: list = field(default_factory=list)
    saturated: np.ndarray = None  # (n, n_condensibles)

    @property
    def p_surf(self):
        return float(self.p_int[0])

    def vmr(self, species, where="lay"):
        j = self.species.index(species)
        return (self.x_lay if where == "lay" else self.x_int)[:, j]

    def column_mass(self, species):
        """Column mass of a species [g/cm^2] (layer mass fractions x M / mu times the layer mass dp / g)."""
        j = self.species.index(species)
        q = self.x_lay[:, j] * self.M[j] / self.mu_lay
        dp = self.p_int[:-1] - self.p_int[1:]
        return float(np.sum(q * dp) / self.g)

    def pressure_equivalent(self, species):
        """Column mass times g [dyn/cm^2]: the surface pressure the species would exert alone (its 'inventory')."""
        return self.column_mass(species) * self.g


class SurfacePressure:
    """Settings and bookkeeping of a variable surface pressure (func None = fixed)."""

    def __init__(self, func=None, update_every=200, relax=0.5, tol=1e-3, p_min=None, p_max=None, max_factor=2.0,
                 settle=None, label=None):
        self.func = func
        self.update_every = int(update_every)
        self.relax = float(relax)
        self.tol = float(tol)
        self.p_min = None if p_min is None else float(p_min)
        self.p_max = None if p_max is None else float(p_max)
        self.max_factor = float(max_factor)
        self.settle = int(settle) if settle is not None else self.update_every
        self.label = label or (getattr(func, "__qualname__", None) or repr(func) if func else "fixed")
        if not 0 < self.relax <= 1 or self.max_factor <= 1 or self.tol <= 0 or self.update_every < 1:
            raise ValueError("surface_pressure: need 0 < relax <= 1, max_factor > 1, tol > 0, update_every >= 1")
        self.reset()

    def reset(self):
        self.converged = self.func is None
        self.last_change_iter = -10**9
        self.history = []          # (iteration, p_surf, p_target)

    @property
    def active(self):
        return self.func is not None

    def step(self, p_now, p_target):
        """Damped, limited, clamped new surface pressure."""
        d = self.relax * (np.log(p_target) - np.log(p_now))
        d = float(np.clip(d, -np.log(self.max_factor), np.log(self.max_factor)))
        p = p_now * np.exp(d)
        if self.p_min is not None:
            p = max(p, self.p_min)
        if self.p_max is not None:
            p = min(p, self.p_max)
        return p

    def evaluate(self, state, iteration):
        """Target and new p_surf for the current state; sets `converged`. Returns the new p_surf or None (no change)."""
        p_now = state.p_surf
        p_target = float(self.func(state))
        if not np.isfinite(p_target) or p_target <= 0:
            raise ValueError(f"surface_pressure function returned {p_target!r}")
        self.history.append((int(iteration), p_now, p_target))
        if abs(np.log(p_target / p_now)) < self.tol:
            self.converged = iteration - self.last_change_iter >= self.settle
            return None
        p_new = self.step(p_now, p_target)
        if abs(np.log(p_new / p_now)) < 1e-12:   # clamped at p_min / p_max
            self.converged = iteration - self.last_change_iter >= self.settle
            if not getattr(self, "_warned_clamp", False):
                log.warning(f"surface pressure target {p_target:.4g} dyn/cm^2 outside [p_min, p_max]; kept at {p_now:.4g}")
                self._warned_clamp = True
            return None
        self.converged = False
        self.last_change_iter = int(iteration)
        return p_new

    def stats(self):
        return {"surface_pressure_function": self.label, "converged": bool(self.converged),
                "history": [[i, p, t] for i, p, t in self.history]}

    def __repr__(self):
        if not self.active:
            return "SurfacePressure(fixed)"
        return (f"SurfacePressure({self.label},every={self.update_every},relax={self.relax:g},tol={self.tol:g},"
                f"p_max={self.p_max})")


# --- example surface-pressure functions (factories, use with `kwargs`) ---

def fixed_inventory(species, p_inventory):
    """
    Keep the column mass of `species` fixed at the inventory p_inventory [dyn/cm^2] (the surface pressure it would
    exert alone, p_a,I of Ding & Pierrehumbert 2016). Other gases (e.g. vapour from an ocean, given as a condensible
    with a large base VMR) add to the surface pressure. Fixed point: p_s * p_inventory / (column mass * g).
    """
    p_inv = float(p_inventory)

    def target(state):
        return state.p_surf * p_inv / state.pressure_equivalent(species)
    target.__qualname__ = f"fixed_inventory({species},{p_inv:g}dyn/cm2)"
    return target


def ocean_plus_inventory(condensible, p_inventory, rh=1.0):
    """p_s = p_inventory + RH p_sat,c(T_surf) [dyn/cm^2]: a dilute-limit ocean over a fixed background (simple test)."""
    from chelio_sim.thermo import Condensible
    c = Condensible(condensible, rh=rh)
    p_inv = float(p_inventory)

    def target(state):
        return p_inv + c.rh * float(c.p_sat(np.array([state.T_surf]))[0])
    target.__qualname__ = f"ocean_plus_inventory({condensible},{p_inv:g}dyn/cm2,RH={rh:g})"
    return target
