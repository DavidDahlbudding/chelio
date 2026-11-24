"""
Column model of the session runner: composition, condensation and convection physics on top of HELIOS.

One ColumnModel per run (helios_session.HeliosSession.run(..., column_model=...)). Data flow, evaluated in this
order (HELIOS layer index 0 = bottom; merged profile = interfaces and layers interleaved bottom -> top):

  surface pressure  (surface.SurfacePressure, every `update_every` convection iterations, damped) -> pressure grid
  base composition  composition(p, T) (composition.py), every 10th iteration (HELIOS' VMR update)
  available vapour  per condensible: where the base composition is already saturated (e.g. an equilibrium grid
                    with local condensation), the vapour a parcel brings from below = the base value at the last
                    unsaturated level below (carried upward); elsewhere the base value
  cold trap         per condensible: x <= running minimum (bottom -> top) of RH p_sat(T) / p; the gas removed is
                    given to the other species (mixfile_utils.redistribute_capped). VMRs for HELIOS' RT.
  thermodynamics    every iteration at the current T (thermo.py): mixture c_p (JANAF), saturation flags, xi,
                    the inhibition parameter Gamma (with hysteresis), and the TARGET lapse rates of HELIOS' half
                    layers: secant exponents of a parcel ascent/descent (RK2) from each layer centre to its two
                    interfaces, dry adiabat where unsaturated, multi-condensible pseudoadiabat where saturated
                    (the parcel saturates inside a half layer at its lifting condensation level); with inhibition in
                    `jump` mode a saturated parcel with Gamma > 1 relaxes to Gamma = 1 over jump_width in ln p
                    (sub-grid inhibited radiative layer = temperature jump at the cloud base, Leconte et al. 2017)
  thresholds        per layer pair (convection.py): target; + max(nabla_mu, 0) for Ledoux in unsaturated pairs;
                    +inf if a layer of the pair is inhibited (inhibition mode `radiative`)
  HELIOS            kappa_lay / kappa_int := targets, c_p_lay := mixture c_p (bypasses the kappa table and the
                    log-T axis of HELIOS' cp_interpol kernel); conv_check / mark / conv_correct use thresholds and
                    targets (convection.py); inhibited layers are never stitched

Saturation is decided with one source, thermo.Condensible (the p_sat fits of mixfile_utils / GGchem).

Units are HELIOS' cgs units: p [dyn/cm^2], T [K], molar c_p [erg mol^-1 K^-1], molar masses [g/mol], g [cm/s^2]
(also in <name>_column.dat and the history). Notation (Ding & Pierrehumbert 2016): a = non-condensible, c = condensible
vapour, l = condensate, sat = saturated; x = mole fraction.
"""
import logging

import numpy as np

from chelio_sim import thermo
from chelio_sim.convection import ConvectionSettings
from chelio_sim.mixfile_utils import redistribute_capped
from chelio_sim.surface import ColumnState, SurfacePressure

log = logging.getLogger(__name__)

NEGLIGIBLE_VMR = 1e-29


def merged_profile(p_lay, p_int, a_lay, a_int):
    """Interleave interface and layer values bottom -> top: int0, lay0, int1, lay1, ..., lay(n-1), int(n)."""
    n = len(p_lay)
    p = np.empty(2 * n + 1)
    p[0::2], p[1::2] = p_int, p_lay
    a = np.empty((2 * n + 1,) + np.shape(a_lay)[1:])
    a[0::2], a[1::2] = a_int, a_lay
    return p, a


class ColumnModel:
    """
    composition           base composition (composition.py; f(p [dyn/cm^2], T [K], species) -> {species: VMR})
    condensibles          thermo.Condensible objects or names (cold trap, moist adiabat, inhibition)
    convection            convection.ConvectionSettings or dict
    surface               surface.SurfacePressure (default: fixed)
    composition_damping   w in [0, 1): log VMRs handed to HELIOS = (1-w) new + w previous (0: none)
    vapour_supply         running_max (default) | saturated | base: vapour of the condensibles available from below
                          (see _available). With an equilibrium (GGchem) grid this is a workaround: condensibles with
                          a GGchem grid are not recommended for now (logged; docs/column_model_handoff.md)
    base_saturation_tol   for vapour_supply saturated: a base VMR counts as saturated if >= (1 - tol) RH x_sat
    """

    def __init__(self, composition, condensibles=(), convection=None, surface=None, composition_damping=0.0,
                 vapour_supply="running_max", base_saturation_tol=0.02):
        self.composition = composition
        self.condensibles = [c if isinstance(c, thermo.Condensible) else thermo.Condensible(c) for c in condensibles]
        if isinstance(convection, ConvectionSettings):
            self.settings = convection
        else:
            self.settings = ConvectionSettings.from_dict(convection)
        self.surface = surface or SurfacePressure()
        self.composition_damping = float(composition_damping)
        if not 0 <= self.composition_damping < 1:
            raise ValueError("composition_damping must be in [0, 1)")
        self.base_saturation_tol = float(base_saturation_tol)
        if vapour_supply not in ("running_max", "saturated", "base"):
            raise ValueError(f"vapour_supply must be running_max, saturated or base, got {vapour_supply!r}")
        self.vapour_supply = vapour_supply
        grid = getattr(composition, "grid", None)
        if self.condensibles and grid is not None and grid.meta.get("kind") == "equilibrium":
            log.warning("Equilibrium (GGchem) chemistry grid with condensibles is NOT RECOMMENDED for now: the interpolated "
                        "grid VMRs of condensibles lie far below saturation at low T, and vapour_supply "
                        f"{self.vapour_supply!r} is only a workaround (see docs/column_model_handoff.md, 'Known problem: "
                        "GGchem grids and condensibles')")
        if self.settings.inhibition and self.condensibles:
            log.warning(f"Condensation-inhibited convection is EXPERIMENTAL (inhibition_mode {self.settings.inhibition_mode}); "
                        "set convection: {inhibition: false} to switch it off")
        if self.settings.thermo_source == "table" and self.condensibles and (self.settings.moist or self.settings.inhibition):
            raise ValueError("convection.thermo_source 'table' is dry only: set moist and inhibition to false "
                             "(or use thermo_source 'host')")

    # --- setup ---
    def bind(self, quant, species, cia_partners):
        """Called once per run after HELIOS' setup (species file read, grid built)."""
        if quant.iso != 0:
            raise ValueError("The column model needs non-isothermal layers (HELIOS iso = 0)")
        if quant.convection != 1:
            log.warning("HELIOS convection is off: the column model only sets the composition")
        self.species = list(species)
        self.index = {s: j for j, s in enumerate(self.species)}
        self.thermo = thermo.MixtureThermo(self.species, self.condensibles)
        self.cond = self.thermo.cond_idx
        self.M_c = self.thermo.M[self.cond]
        self.cia = dict(cia_partners)
        self.g = float(quant.g)
        self.dirty = True
        self.in_loop = False
        self.x_avail = None
        self.x_prev = None
        self.inhibited = np.zeros(quant.nlayer, bool)
        self.thr_lay = np.array(quant.kappa_lay, float)
        self.thr_int = np.array(quant.kappa_int, float)
        self.sat_pair, self.sat_surf = None, False
        self.conv_prev = None
        self.n_unfrozen = 0
        self.diag = None
        self.adjust_cap_hits = 0
        self.last_fudge = []
        self.n_composition_updates = 0
        self.n_thermo_updates = 0
        self.history = []
        self.regrids = []
        self.surface.reset()

    def mark_dirty(self):
        self.dirty = True

    # --- composition ---
    def _base(self, p_dyn, T):
        """Base mole fractions normalised over the active species, and their original sum (<= 1; e.g. GGchem
        VMRs of the active species sum to slightly less than 1, the rest are species HELIOS does not know).
        Thermodynamics use the normalised values (as HELIOS' mean molecular mass); the VMRs handed to the RT are
        scaled back by the sum."""
        out = self.composition(p_dyn, T, self.species)
        x = np.full((len(p_dyn), len(self.species)), NEGLIGIBLE_VMR)
        for s, v in out.items():
            if s in self.index:
                x[:, self.index[s]] = np.clip(np.asarray(v, float), NEGLIGIBLE_VMR, None)
        total = x.sum(axis=1, keepdims=True)
        self._base_sum = np.minimum(total[:, 0], 1.0)
        return x / total

    def _available(self, x_base, T, p_dyn):
        """
        Vapour of the condensibles available to a parcel coming from below (vapour_supply):
          running_max  running maximum of the base VMR from the bottom up: a decrease with height is taken as
                       condensation (equilibrium grids condense locally, and their interpolated VMRs lie below the
                       saturation curve of thermo.Condensible), which the cold trap then redoes consistently
          saturated    carry the base value of the last level below where the base is not saturated within
                       base_saturation_tol (keeps chemical decreases with height)
          base         the base VMRs as they are
        """
        x = x_base.copy()
        if not len(self.cond) or self.vapour_supply == "base":
            return x
        if self.vapour_supply == "running_max":
            x[:, self.cond] = np.maximum.accumulate(x_base[:, self.cond], axis=0)
            return x / x.sum(axis=1, keepdims=True)
        x_sat, _ = self.thermo.saturation(T, p_dyn)
        n = len(T)
        for k, j in enumerate(self.cond):
            saturated = x_base[:, j] >= (1 - self.base_saturation_tol) * x_sat[:, k]
            saturated[0] = False
            src = np.maximum.accumulate(np.where(saturated, 0, np.arange(n)))
            x[:, j] = x_base[src, j]
        return x / x.sum(axis=1, keepdims=True)

    def _cold_trap(self, x_av, T, p_dyn):
        """(x, x_sat, xi, saturated) on a bottom -> top profile; x_sat, xi, saturated: (n, K) for the condensibles."""
        x = x_av.copy()
        K = len(self.cond)
        if not K:
            z = np.zeros((len(T), 0))
            return x, z, z, z.astype(bool)
        x_sat, xi = self.thermo.saturation(T, p_dyn)
        caps = np.full_like(x, np.inf)
        caps[:, self.cond] = np.minimum.accumulate(x_sat, axis=0)
        x = redistribute_capped(x, caps)
        sat = np.isfinite(x_sat) & (x[:, self.cond] >= (1 - 1e-9) * x_sat)
        pv = self.settings.pure_vapour
        if pv is not None:
            sat |= x[:, self.cond] >= pv   # pure vapour: on the dewpoint adiabat even if superheated
        return x, x_sat, xi, sat

    def composition_profile(self, p_dyn, T):
        """(base, available, cold-trapped) mole fractions (normalised) on a bottom -> top profile."""
        x_base = self._base(p_dyn, T)
        x_av = self._available(x_base, T, p_dyn)
        return x_base, x_av, self._cold_trap(x_av, T, p_dyn)[0]

    def vmr_profile(self, p_dyn, T):
        """Cold-trapped VMRs (n, n_species) along a profile (p [dyn/cm^2]) ordered bottom -> top (for vertical_mix
        files), scaled like the VMRs handed to HELIOS."""
        x = self.composition_profile(np.asarray(p_dyn, float), np.asarray(T, float))[2]
        return x * self._base_sum[:, None]

    def update_composition(self, quant):
        """HELIOS' VMR update (every 10th iteration): VMRs of all species (+ CIA products) on layers and interfaces."""
        quant.T_lay = quant.dev_T_lay.get()
        quant.T_int = quant.dev_T_int.get()
        quant.p_lay = quant.dev_p_lay.get()
        quant.p_int = quant.dev_p_int.get()
        n = quant.nlayer
        p, T = merged_profile(quant.p_lay, quant.p_int, quant.T_lay[:n], quant.T_int)
        _, x_av, x = self.composition_profile(p, T)
        w = self.composition_damping
        if w > 0 and self.x_prev is not None and self.x_prev.shape == x.shape:
            x = np.exp((1 - w) * np.log(x) + w * np.log(self.x_prev))
            x /= x.sum(axis=1, keepdims=True)
        self.x_prev = x
        self.x_avail = x_av
        x = x * self._base_sum[:, None]
        x_lay, x_int = x[1::2], x[0::2]
        for sp in quant.species_list:
            partners = self.cia.get(sp.name)
            if partners:
                a, b = self.index[partners[0]], self.index[partners[1]]
                lay, itf = x_lay[:, a] * x_lay[:, b], x_int[:, a] * x_int[:, b]
            elif sp.name in self.index:
                lay, itf = x_lay[:, self.index[sp.name]], x_int[:, self.index[sp.name]]
            else:
                continue
            sp.vmr_layer = np.array(lay, quant.fl_prec)
            sp.vmr_interface = np.array(itf, quant.fl_prec)
        self.n_composition_updates += 1
        if not self.in_loop:  # inside the convection loop the session marks the model dirty once per iteration
            self.dirty = True

    # --- thermodynamics and convection arrays ---
    def update_thermo(self, quant, compute_table=None):
        """
        Targets (kappa_lay/int), c_p_lay, thresholds and the inhibited mask at the current temperatures; once per
        convection iteration (helios_session marks the model dirty at the start of each iteration), so the
        marking after the adjustment uses exactly the exponents the adjustment used. compute_table: callable
        running HELIOS' table interpolation (thermo_source 'table').
        """
        if not self.dirty:
            return
        if self.x_avail is None:
            self.update_composition(quant)
        n = quant.nlayer
        T_full = quant.dev_T_lay.get()
        T_lay = np.array(T_full[:n], float)
        T_int = np.array(quant.dev_T_int.get(), float)
        p_lay = np.array(quant.dev_p_lay.get(), float)
        p_int = np.array(quant.dev_p_int.get(), float)
        p, T = merged_profile(p_lay, p_int, T_lay, T_int)
        x, x_sat, xi, sat = self._cold_trap(self.x_avail, T, p)
        x_lay, x_int = x[1::2], x[0::2]
        sat_lay, sat_int, xi_lay, x_sat_lay = sat[1::2], sat[0::2], xi[1::2], x_sat[1::2]
        mu = self.thermo.mu(x_lay)
        mu_int0 = float(self.thermo.mu(x_int[:1])[0])
        K = len(self.cond)

        if self.settings.thermo_source == "table":
            compute_table(quant)
            k_lay = np.array(quant.dev_kappa_lay.get(), float)
            k_int = np.array(quant.dev_kappa_int.get(), float)
            cp = np.array(quant.dev_c_p_lay.get(), float)
            gamma = np.zeros(n)
        else:
            if self.settings.kappa is not None:
                # constant dry adiabat (HELIOS' numeric `kappa value`): c_p = R / kappa everywhere
                kap = float(self.settings.kappa)
                cp = np.full(n, thermo.R / kap)
                k_lay, k_int = np.full(n, kap), np.full(n + 1, kap)
            else:
                cp_s = self.thermo.cp_species(T_lay)
                cp = self.thermo.cp_mix(x_lay, cp_s)
                k_lay, k_int = self._targets(T_lay, p_lay, p_int, x_lay, sat_lay, self.x_avail[1::2], cp_s, cp)
            if K:
                gamma = thermo.inhibition_gamma(x_lay[:, self.cond], xi_lay, sat_lay, self.M_c, mu)
                if self.settings.pure_vapour is not None:   # no background gas left: no mu gradient
                    gamma = np.where((x_lay[:, self.cond] >= self.settings.pure_vapour).any(axis=1), 0.0, gamma)
            else:
                gamma = np.zeros(n)
            quant.kappa_lay = np.array(k_lay, quant.fl_prec)
            quant.kappa_int = np.array(k_int, quant.fl_prec)
            quant.c_p_lay = np.array(cp, quant.fl_prec)
            quant.dev_kappa_lay.set(quant.kappa_lay)
            quant.dev_kappa_int.set(quant.kappa_int)
            quant.dev_c_p_lay.set(quant.c_p_lay)

        sat_any = sat_lay.any(axis=1) if K else np.zeros(n, bool)
        if self.settings.inhibition and K and self.settings.inhibition_mode == "radiative":
            h = self.settings.inhibition_hysteresis
            inh = sat_any & ((gamma > 1 + h) | (self.inhibited & (gamma > 1 - h)))
        else:
            inh = np.zeros(n, bool)
        self.inhibited = inh
        sat_any_int = sat_int.any(axis=1) if K else np.zeros(n + 1, bool)
        self._thresholds(k_lay, k_int, mu, mu_int0, p_lay, p_int, sat_any, sat_any_int, inh)
        self.diag = {"T": T_lay, "T_surf": float(T_full[n]), "T_int": T_int, "p_lay": p_lay, "p_int": p_int,
                     "x_lay": x_lay, "x_int": x_int, "mu": mu, "cp": cp, "kappa_lay": k_lay, "kappa_int": k_int,
                     "gamma": gamma, "saturated": sat_lay, "x_sat_lay": x_sat_lay, "xi": xi_lay}
        self.n_thermo_updates += 1
        self.dirty = False

    def _targets(self, T_lay, p_lay, p_int, x_lay, sat_lay, x_avail_lay, cp_s, cp):
        """Secant exponents of the parcel lapse rate over the upper (kappa_lay) and lower (kappa_int) half layers."""
        n = len(T_lay)
        K = len(self.cond)
        if not K or not self.settings.moist:
            k = thermo.R / cp
            return k, np.append(k, k[-1])
        cond = self.cond
        x_a_norm = x_lay.copy()
        x_a_norm[:, cond] = 0.0
        s = x_a_norm.sum(axis=1, keepdims=True)
        x_a_norm = np.where(s > 0, x_a_norm / np.where(s > 0, s, 1.0), 0.0)
        with np.errstate(invalid="ignore"):
            cp_a = self.thermo.cp_mix(x_a_norm, cp_s)
        cp_a = np.where(np.isfinite(cp_a), cp_a, 0.0)
        cp_c = cp_s[:, cond]
        x_c_lay = x_lay[:, cond]
        x_c_up = x_c_lay
        x_c_dn = np.where(sat_lay.any(axis=1, keepdims=True), np.maximum(x_avail_lay[:, cond], x_c_lay), x_c_lay)
        multi = self.settings.multi
        M_a = x_a_norm @ self.thermo.M
        M_c = self.M_c
        jump = self.settings.inhibition and self.settings.inhibition_mode == "jump"

        def state(lnT, lnp, x_c_tot):
            x_sat, xi = self.thermo.saturation(np.exp(lnT), np.exp(lnp))
            # a parcel that starts at saturation (x_c_tot = x_sat of the layer) is saturated: without the tolerance
            # the comparison flips at rounding level and the target jumps (dry first RK stage)
            sat_c = x_sat <= x_c_tot * (1 + 1e-6)
            x_c = np.where(sat_c, x_sat, x_c_tot)
            if pv is not None:
                # pure vapour (convection.pure_vapour): saturated even if superheated, x_c stays at its total
                sat_c = sat_c | (x_c_tot >= pv)
            x_a = np.clip(1.0 - x_c.sum(axis=1), 0.0, 1.0)
            return x_sat, xi, sat_c, x_c, x_a

        width = float(self.settings.jump_width)
        pv = self.settings.pure_vapour

        def grad(lnT, lnp, x_c_tot):
            x_sat, xi, sat_c, x_c, x_a = state(lnT, lnp, x_c_tot)
            cpm = x_a * cp_a + np.sum(x_c * cp_c, axis=1)
            g = thermo.moist_gradient(x_c, xi, sat_c, cpm, multi)
            if jump:
                # sub-grid inhibited layer: a saturated parcel with Gamma > 1 relaxes to Gamma = 1 over `width` in
                # ln p (extra gradient (Gamma - 1) / (width dGamma/dlnT)); the same ODE up and down
                mu = x_a * M_a + x_c @ M_c
                x_c_unsat = np.where(sat_c, 0.0, x_c)
                with np.errstate(invalid="ignore", divide="ignore"):
                    M_unsat = (x_a * M_a + x_c_unsat @ M_c) / (x_a + x_c_unsat.sum(axis=1))
                    sat_g = sat_c if pv is None else sat_c & ~(x_c >= pv).any(axis=1, keepdims=True)  # pure vapour: no jump
                    w = np.where(sat_g, x_c * xi * (M_c[None, :] - M_unsat[:, None]), 0.0) / mu[:, None]
                gam = np.nansum(w, axis=1)
                dgam = np.nansum(w * xi, axis=1)  # dGamma/dlnT (x_sat ~ exp(int xi dlnT))
                with np.errstate(invalid="ignore", divide="ignore"):
                    extra = np.where((gam > 1.0) & (dgam > 0), (gam - 1.0) / (width * dgam), 0.0)
                g = g + np.clip(np.nan_to_num(extra), 0.0, 10.0)
            return g

        def integrate(lnp_end, x_c_tot):
            nsub = int(self.settings.parcel_substeps)
            lnT = np.log(T_lay)
            lnp = np.log(p_lay)
            h = (lnp_end - lnp) / nsub
            for _ in range(nsub):
                k1 = grad(lnT, lnp, x_c_tot)
                k2 = grad(lnT + 0.5 * h * k1, lnp + 0.5 * h, x_c_tot)
                lnT = lnT + h * k2
                lnp = lnp + h
            return lnT

        lnT0 = np.log(T_lay)
        lnT_up = integrate(np.log(p_int[1:]), x_c_up)
        lnT_dn = integrate(np.log(p_int[:-1]), x_c_dn)
        k_lay = (lnT_up - lnT0) / (np.log(p_int[1:]) - np.log(p_lay))
        k_int = (lnT0 - lnT_dn) / (np.log(p_lay) - np.log(p_int[:-1]))
        return k_lay, np.append(k_int, k_lay[-1])

    def _thresholds(self, k_lay, k_int, mu, mu_int0, p_lay, p_int, sat_any, sat_any_int, inh):
        n = len(k_lay)
        thr_lay = np.array(k_lay, float)
        thr_int = np.array(k_int, float)
        i = np.arange(n - 1)
        moist = self.settings.moist and len(self.cond) > 0
        sat_pair = (sat_any[i] | sat_any[i + 1] | sat_any_int[i + 1]) if moist else np.zeros(n - 1, bool)
        sat_surf = bool(sat_any[0] or sat_any_int[0]) if moist else False
        self.sat_pair, self.sat_surf = sat_pair, sat_surf  # moist targets (maintenance tolerance in mark)
        self.nabla_mu = np.zeros(n)
        if self.settings.criterion == "ledoux":
            dmu = np.log(mu[i + 1] / mu[i]) / np.log(p_lay[i + 1] / p_lay[i])
            dmu_s = np.log(mu[0] / mu_int0) / np.log(p_lay[0] / p_int[0])
            self.nabla_mu[:-1] = dmu
            add = np.where(sat_pair, 0.0, np.maximum(dmu, 0.0))
            thr_lay[i] += add
            thr_int[i + 1] += add
            if not sat_surf:
                thr_int[0] += max(dmu_s, 0.0)
        if self.settings.inhibition and np.any(inh):
            pair_inh = inh[i] | inh[i + 1]
            thr_lay[i[pair_inh]] = np.inf
            thr_int[i[pair_inh] + 1] = np.inf
            if inh[0]:
                thr_int[0] = np.inf
        thr_lay = np.where(np.isnan(thr_lay), np.inf, thr_lay)
        thr_int = np.where(np.isnan(thr_int), np.inf, thr_int)
        self.thr_lay = np.maximum(thr_lay, k_lay)
        self.thr_int = np.maximum(thr_int, k_int)

    # --- state, bookkeeping, output ---
    def state(self, quant):
        d = self.diag
        K = len(self.cond)
        return ColumnState(p_lay=d["p_lay"], p_int=d["p_int"], T_lay=d["T"], T_int=d["T_int"], T_surf=d["T_surf"],
                           species=self.species, x_lay=d["x_lay"], x_int=d["x_int"], M=self.thermo.M,
                           mu_lay=d["mu"], g=self.g, conv_layer=np.array(quant.conv_layer, int),
                           inhibited=self.inhibited.copy(), condensibles=[c.name for c in self.condensibles],
                           saturated=d["saturated"] if K else None)

    def record(self, quant):
        d = self.diag or {}
        sat = d.get("saturated")
        self.history.append([int(quant.iter_value), float(quant.p_int[0]), int(self.inhibited.sum()),
                             int(sat.any(axis=1).sum()) if sat is not None and sat.size else 0, int(self.adjust_cap_hits),
                             np.flatnonzero(self.inhibited).tolist(), list(getattr(self, "last_fudge", []))])

    def write_column_file(self, path, quant):
        """Per-layer diagnostics of the final state (bottom -> top), see the header."""
        d = self.diag
        n = quant.nlayer
        conv = np.array(quant.conv_layer, int)
        cols = {"layer": np.arange(n), "p[dyn/cm2]": d["p_lay"], "T[K]": d["T"], "mu[g/mol]": d["mu"],
                "c_p[erg/mol/K]": d["cp"], "nabla_ad": thermo.R / d["cp"], "kappa_lower": d["kappa_int"][:n],
                "kappa_upper": d["kappa_lay"], "thr_lower": self.thr_int[:n], "thr_upper": self.thr_lay,
                "nabla_mu": getattr(self, "nabla_mu", np.zeros(n)), "Gamma": d["gamma"],
                "inhibited": self.inhibited.astype(int), "conv": conv[:n]}
        for k, c in enumerate(self.condensibles):
            cols[f"x_{c.name}"] = d["x_lay"][:, self.cond[k]]
            cols[f"S_{c.name}"] = d["x_lay"][:, self.cond[k]] / d["x_sat_lay"][:, k]
            cols[f"sat_{c.name}"] = d["saturated"][:, k].astype(int)
        names = list(cols)
        data = np.column_stack([np.asarray(cols[k], float) for k in names])
        with open(path, "w") as f:
            f.write(f"# column model {self!r}\n")
            f.write(f"# surface: p = {d['p_int'][0]:.8e} dyn/cm2, T = {d['T_surf']:.6f} K, conv = {conv[n]}. "
                    "kappa_lower/upper: target dlnT/dlnp over the lower/upper half layer; thr_*: onset threshold of "
                    "the pair this half belongs to (inf: inhibited); nabla_mu: pair (i, i+1); S = x / (RH x_sat)\n")
            f.write("".join(f"{k:<16}" for k in names).rstrip() + "\n")
            for row in data:
                f.write("".join(f"{v:<16.8e}" if np.isfinite(v) or np.isinf(v) else f"{'nan':<16}" for v in row).rstrip() + "\n")

    def stats(self):
        out = {"column_model": repr(self), "history_columns": ["iteration", "p_surf[dyn/cm2]", "n_inhibited", "n_saturated",
                                                               "adjust_cap_hits", "inhibited_layers", "fudge_per_zone"],
               "composition_updates": self.n_composition_updates,
               "thermo_updates": self.n_thermo_updates, "adjust_cap_hits": self.adjust_cap_hits,
               "unfrozen_layers": self.n_unfrozen,
               "inhibited_layers_final": np.flatnonzero(self.inhibited).tolist(),
               "history": self.history, "regrids": self.regrids}
        if self.diag is not None and len(self.cond):
            sat = self.diag["saturated"]
            out["saturated_layers_final"] = {c.name: np.flatnonzero(sat[:, k]).tolist() for k, c in enumerate(self.condensibles)}
        if hasattr(self.composition, "stats"):
            out["composition"] = self.composition.stats()
        if self.surface.active:
            out["surface"] = self.surface.stats()
        return out

    def __repr__(self):
        cond = ",".join(repr(c) for c in self.condensibles) or "none"
        extra = f",damping={self.composition_damping:g}" if self.composition_damping else ""
        if self.vapour_supply != "running_max":
            extra += f",supply={self.vapour_supply}"
        return (f"ColumnModel({self.composition!r}; condensibles={cond}; {self.settings.label()}; "
                f"{self.surface!r}{extra})")
