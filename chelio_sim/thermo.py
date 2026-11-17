"""
Thermodynamics of condensing atmospheres for the session runner's column model (chelio_sim.column).

Units: cgs as in HELIOS - p [dyn/cm^2], T [K], molar c_p [erg mol^-1 K^-1] (as HELIOS' c_p_lay), R [erg mol^-1 K^-1],
molar masses M [g/mol], latent heats L [erg/g].

Notation (Ding & Pierrehumbert 2016): subscript a = non-condensible ("air"; with several condensibles: all species
that are not saturated at that point), c = condensible (vapour), l = condensate (liquid or solid), sat = saturated;
x = mole fraction, q = mass mixing ratio (mass fraction), r = mixing ratio relative to a; index k numbers the
condensibles.

Contents
  Condensible       saturation vapour pressure p_sat(T) per phase branch, its analytic logarithmic derivative
                    xi = dln p_sat / dln T, the latent heat L = R T xi / M_c (Clausius-Clapeyron, as Leconte et al.
                    2017, Sect. 4.1) and a relative humidity RH (the cap is RH p_sat).
  MixtureThermo     JANAF c_p of the gas species, molar masses (HELIOS species_lib), dry adiabatic gradient R / c_p,
                    saturation mole fractions x_sat = RH p_sat / p.

Multi-condensible pseudoadiabat (ideal gases, condensate removed; x_c,k mole fractions of the saturated condensibles,
x_a = 1 - sum_k x_c,k, c_p = sum over all species of x_i c_p,i):

    nabla* = R (x_a + S1) / (x_a c_p + R (x_a S2 + S1^2)),   S1 = sum_k x_c,k xi_k,   S2 = sum_k x_c,k xi_k^2

For one condensible this is exactly Leconte et al. (2017) eq. 12 (= Ding & Pierrehumbert 2016 eq. 12 in total
pressure): nabla* = R (1 + r_c xi) / (c_p + R r_c xi^2), r_c = x_c / x_a. Without saturated species it is R / c_p;
for a pure condensible it is 1 / xi (Clausius-Clapeyron). `python3 -m chelio_sim.thermo --check` tests these limits.

Inhibition of moist convection (Leconte et al. 2017, eqs. 13-17, 45, B.6): a saturated layer is stable against
moist convection if

    Gamma = sum_k x_c,k xi_k (M_c,k - M_a) / mu > 1,

M_a = mean molar mass of the species that are not saturated, mu = mean molar mass. For one condensible
Gamma = varpi q_c xi (= alpha_mu gamma_s of the paper), so Gamma > 1 <=> q_c > q_cri = R T / (varpi M_c L),
varpi = (M_c - M_a) / M_c. Gamma < 0 for condensibles lighter than the background (H2O in N2): never inhibited.
The sum over several condensibles generalises eq. 13 (exact for one species).

Saturation-pressure branches: the formulas of mixfile_utils.p_sat (= GGchem DustChem fits), with
  - a solid CO2 branch below the triple point (Fray & Schmitt 2009, Planet. Space Sci. 57, 2053, 40-195 K),
  - no condensation above the critical temperature (p_sat = inf),
  - below `t_floor` (half the triple-point temperature) a Clausius-Clapeyron extrapolation with constant L
    (the Antoine fits diverge at T = -C).
N2, CO, CH4 use their liquid fits below the triple point too (as GGchem does).
"""
import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)

R = 8.314462618e7        # erg mol^-1 K^-1
BAR = 1.0e6              # dyn/cm^2
MMHG = 1.3328e3          # dyn/cm^2 (value used by GGchem and mixfile_utils)
LN10 = np.log(10.0)
J_TO_ERG = 1.0e7


# --- saturation-pressure branches: (ln p [dyn/cm^2], xi = dln p / dln T) ---

def _antoine10(A, B, C):
    """log10(p / bar) = A - B / (T + C)"""
    def f(T):
        return LN10 * (A - B / (T + C)) + np.log(BAR), LN10 * B * T / (T + C) ** 2
    return f


def _yaws(C0, C1, C2, C3, C4):
    """log10(p / mmHg) = C0 + C1/T + C2 log10 T + C3 T + C4 T^2"""
    def f(T):
        lnp = LN10 * (C0 + C1 / T + C2 * np.log10(T) + C3 * T + C4 * T ** 2) + np.log(MMHG)
        xi = LN10 * (-C1 / T + C3 * T + 2.0 * C4 * T ** 2) + C2
        return lnp, xi
    return f


def _inverse_poly_ln_bar(*a):
    """ln(p / bar) = sum_i a_i T^-i"""
    def f(T):
        lnp = sum(ai * T ** (-i) for i, ai in enumerate(a)) + np.log(BAR)
        xi = sum(-i * ai * T ** (-i) for i, ai in enumerate(a))
        return lnp, xi
    return f


def _h2o_ice_am01(T):
    """Ackerman & Marley (2001) ice: p = 6111.5 exp((23.036 TC - TC^2/333.7) / (TC + 279.82)) dyn/cm^2"""
    a, b, c = 23.036, 333.7, 279.82
    TC = np.minimum(2000.0, T) - 273.15
    num = a * TC - TC ** 2 / b
    den = TC + c
    lnp = np.log(6111.5) + num / den
    dfdT = ((a - 2.0 * TC / b) * den - num) / den ** 2
    return lnp, T * dfdT


# species: molar mass from HELIOS (species_lib) at runtime; here: critical T, triple T, branches (T_upper, func)
# branches are ordered by T; a branch applies below its T_upper (the last one up to T_crit)
CONDENSIBLE_DATA = {
    "H2O": {"T_crit": 647.1, "T_triple": 273.16,
            "branches": [(273.16, _h2o_ice_am01),
                         (np.inf, _yaws(29.8605, -3.1522e3, -7.3037, 2.4247e-9, 1.8090e-6))]},
    "CO2": {"T_crit": 304.19, "T_triple": 216.58,
            "branches": [(216.58, _inverse_poly_ln_bar(1.476e1, -2.571e3, -7.781e4, 4.325e6, -1.207e8, 1.350e9)),
                         (np.inf, _yaws(35.0187, -1.5119e3, -11.335, 9.3383e-3, 7.7626e-10))]},
    "CH4": {"T_crit": 190.8, "T_triple": 90.68, "branches": [(np.inf, _antoine10(3.9895, 443.028, -0.49))]},
    "NH3": {"T_crit": 405.5, "T_triple": 195.4, "branches": [(np.inf, _inverse_poly_ln_bar(10.53, -2161.0, -86596.0))]},
    "N2": {"T_crit": 126.2, "T_triple": 63.18, "branches": [(np.inf, _antoine10(3.7362, 264.651, -6.788))]},
    "CO": {"T_crit": 133.16, "T_triple": 68.1,
           "branches": [(np.inf, _yaws(51.8145, -7.8824e2, -22.734, 5.1225e-2, 4.6603e-11))]},
    "O2": {"T_crit": 154.33, "T_triple": 54.36, "branches": [(np.inf, _antoine10(3.9523, 340.024, -4.144))]},
    "H2": {"T_crit": 33.2, "T_triple": 13.8033, "branches": [(np.inf, _antoine10(3.54314, 99.395, 7.726))]},
}
# species that "condensibles: auto" picks from the active species (H2 only condenses below 33 K)
AUTO_CONDENSIBLES = ("H2O", "CO2", "CH4", "NH3", "N2", "CO", "O2")


def molar_mass(name):
    from chelio_sim.mixfile_utils import species_lib
    return float(species_lib[name].weight)


class Condensible:
    """
    One condensing species. p_sat / xi are vectorised over T (any shape). RH scales the saturation cap.
    """

    def __init__(self, name, rh=1.0, M=None):
        if name not in CONDENSIBLE_DATA:
            raise ValueError(f"No saturation data for {name!r} (known: {sorted(CONDENSIBLE_DATA)})")
        self.name = name
        self.rh = float(rh)
        if not 0 < self.rh <= 1:
            raise ValueError(f"{name}: relative humidity must be in (0, 1], got {rh}")
        d = CONDENSIBLE_DATA[name]
        self.T_crit = d["T_crit"]
        self.T_triple = d["T_triple"]
        self.branches = d["branches"]
        self.t_floor = 0.5 * self.T_triple
        self.M = float(M) if M is not None else molar_mass(name)
        self._floor = self._eval_branches(np.array([self.t_floor]))

    def _eval_branches(self, T):
        lnp = np.empty_like(T)
        xi = np.empty_like(T)
        lower = -np.inf
        for upper, func in self.branches:
            m = (T >= lower) & (T < upper)
            if np.any(m):
                lnp[m], xi[m] = func(T[m])
            lower = upper
        return lnp, xi

    def ln_psat_xi(self, T):
        """(ln p_sat [dyn/cm^2], xi); ln p_sat = +inf and xi = 0 above the critical temperature."""
        T = np.asarray(T, float)
        shape = T.shape
        T = np.atleast_1d(T).ravel()
        Te = np.maximum(T, self.t_floor)
        lnp, xi = self._eval_branches(Te)
        cold = T < self.t_floor
        if np.any(cold):  # Clausius-Clapeyron with L(t_floor): ln p = ln p_f - xi_f t_f (1/T - 1/t_f)
            lnp_f, xi_f = self._floor[0][0], self._floor[1][0]
            lnp[cold] = lnp_f - xi_f * self.t_floor * (1.0 / T[cold] - 1.0 / self.t_floor)
            xi[cold] = xi_f * self.t_floor / T[cold]
        hot = T >= self.T_crit
        lnp[hot] = np.inf
        xi[hot] = 0.0
        return lnp.reshape(shape), xi.reshape(shape)

    def p_sat(self, T):
        return np.exp(self.ln_psat_xi(T)[0])

    def xi(self, T):
        return self.ln_psat_xi(T)[1]

    def latent_heat(self, T):
        """L [erg/g] = R T xi / M_c"""
        T = np.asarray(T, float)
        return R * T * self.xi(T) / self.M

    def x_sat(self, T, p_dyn):
        """Saturation mole fraction x_sat = RH p_sat / p, p [dyn/cm^2] (inf above T_crit)."""
        with np.errstate(over="ignore"):
            return self.rh * self.p_sat(T) / np.asarray(p_dyn, float)

    def __repr__(self):
        return f"{self.name}(RH={self.rh:g})" if self.rh != 1 else self.name


# --- mixtures ---

class MixtureThermo:
    """
    Thermodynamics of the gas mixture of `species` (HELIOS names, no CIA) with `condensibles` among them.
    Mole-fraction arrays have shape (n, n_species), columns in the order of `species`.
    """

    def __init__(self, species, condensibles=()):
        from chelio_sim.mixfile_utils import _get_janaf_cp_interpolator
        self.species = list(species)
        self.M = np.array([molar_mass(s) for s in self.species])
        self.condensibles = list(condensibles)
        missing = [c.name for c in self.condensibles if c.name not in self.species]
        if missing:
            raise ValueError(f"Condensibles {missing} are not among the active species {self.species}")
        self.cond_idx = np.array([self.species.index(c.name) for c in self.condensibles], int)
        self._cp = []
        self.no_cp = []
        for s in self.species:
            try:
                self._cp.append(_get_janaf_cp_interpolator(s))
            except Exception:
                self._cp.append(None)
                self.no_cp.append(s)
        self._warned_cp = False

    # -- per-species --
    def cp_species(self, T):
        """Molar c_p [erg mol^-1 K^-1], shape (n, n_species); NaN for species without JANAF data."""
        T = np.atleast_1d(np.asarray(T, float))
        out = np.full((T.size, len(self.species)), np.nan)
        for j, f in enumerate(self._cp):
            if f is not None:
                out[:, j] = f(T) * J_TO_ERG
        return out

    def cp_mix(self, x, cp_s):
        """Mixture c_p from mole fractions; species without c_p data are left out (renormalised)."""
        have = np.isfinite(cp_s)
        if self.no_cp and not self._warned_cp:
            lost = [s for j, s in enumerate(self.species) if s in self.no_cp and np.nanmax(x[:, j]) > 1e-6]
            if lost:
                log.warning(f"No JANAF c_p for {lost} (VMR > 1e-6): left out of the mixture c_p")
            self._warned_cp = True
        w = np.where(have, x, 0.0)
        return np.sum(w * np.where(have, cp_s, 0.0), axis=1) / np.sum(w, axis=1)

    def mu(self, x):
        """Mean molar mass [g/mol] (normalised over the given mole fractions)."""
        return x @ self.M / x.sum(axis=1)

    # -- condensibles --
    def saturation(self, T, p_dyn):
        """(x_sat (n, K), xi (n, K)) of the condensibles at (T [K], p [dyn/cm^2])."""
        T = np.atleast_1d(np.asarray(T, float))
        p = np.atleast_1d(np.asarray(p_dyn, float))
        K = len(self.condensibles)
        x_sat = np.empty((T.size, K))
        xi = np.empty((T.size, K))
        for k, c in enumerate(self.condensibles):
            lnp, xik = c.ln_psat_xi(T)
            with np.errstate(over="ignore"):
                x_sat[:, k] = c.rh * np.exp(lnp) / p
            xi[:, k] = xik
        return x_sat, xi


def moist_gradient(x_c, xi, sat, cp, multi="sum"):
    """
    Pseudoadiabatic dlnT/dlnp (see module docstring). x_c, xi, sat: (n, K) mole fractions, xi and saturation flags of
    the condensibles; cp: (n,) mixture molar c_p [erg mol^-1 K^-1]. Equals R/cp where nothing is saturated.
    multi = "sum" (all saturated species) or "dominant" (only the saturated species with the largest x_c xi; debugging).
    """
    x_csat = np.where(sat, x_c, 0.0)
    if multi == "dominant" and x_csat.shape[1] > 1:
        keep = np.argmax(x_csat * xi, axis=1)
        mask = np.zeros_like(sat)
        mask[np.arange(len(keep)), keep] = True
        x_csat = np.where(mask, x_csat, 0.0)
    elif multi != "sum" and multi != "dominant":
        raise ValueError(f"multi must be 'sum' or 'dominant', got {multi!r}")
    S1 = np.sum(x_csat * xi, axis=1)
    S2 = np.sum(x_csat * xi ** 2, axis=1)
    x_a = np.clip(1.0 - np.sum(x_csat, axis=1), 0.0, 1.0)
    num = R * (x_a + S1)
    den = x_a * cp + R * (x_a * S2 + S1 ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        grad = num / den
    dry = R / cp
    grad = np.where(np.isfinite(grad), grad, dry)
    return np.minimum(grad, dry)


def inhibition_gamma(x_c, xi, sat, M_c, mu):
    """
    Gamma = sum_k x_c,k xi_k (M_c,k - M_a) / mu over saturated condensibles (Leconte et al. 2017: Gamma > 1 inhibits
    moist convection). x_c, xi, sat: (n, K); M_c: (K,) molar masses [g/mol]; mu: (n,) mean molar mass [g/mol].
    """
    x_csat = np.where(sat, x_c, 0.0)
    x_a = 1.0 - np.sum(x_csat, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        M_a = (mu - x_csat @ M_c) / x_a
    M_a = np.where(x_a > 1e-12, M_a, 0.0)
    gamma = np.sum(x_csat * xi * (M_c[None, :] - M_a[:, None]), axis=1) / mu
    return np.where(x_a > 1e-12, gamma, 0.0)


def q_critical(c, T, M_a):
    """Critical mass mixing ratio q_cri = R T / (varpi M_c L) (Leconte et al. 2017 eq. 17), varpi = (M_c - M_a) / M_c."""
    varpi = (c.M - M_a) / c.M
    with np.errstate(divide="ignore"):
        return R * T / (varpi * c.M * c.latent_heat(T))  # inf above T_crit (L = 0)


# --- self-check ---

def _check(verbose=True):
    """Tests of the formulas against the papers' limits (python3 -m chelio_sim.thermo --check)."""
    from chelio_sim import mixfile_utils
    ok = True

    def report(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose or not cond:
            print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

    # 1. p_sat agrees with mixfile_utils.p_sat inside the fits' ranges; xi equals the numerical derivative
    ranges = {"H2O": (150, 640), "CO2": (217, 300), "CH4": (92, 188), "NH3": (200, 400), "N2": (64, 125),
              "CO": (69, 130), "O2": (56, 150)}
    for name, (t0, t1) in ranges.items():
        c = Condensible(name)
        T = np.linspace(t0, t1, 400)
        ref = mixfile_utils.p_sat(T, name)
        rel = np.max(np.abs(c.p_sat(T) / ref - 1))
        report(f"p_sat {name} = mixfile_utils.p_sat ({t0}-{t1} K)", rel < 1e-10, f"max rel. diff {rel:.1e}")
        Tn = T[(np.abs(T - c.T_triple) > 1.0)]
        h = 1e-4
        num = (np.log(c.p_sat(Tn * (1 + h))) - np.log(c.p_sat(Tn * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
        rel = np.max(np.abs(c.xi(Tn) / num - 1))
        report(f"xi {name} analytic = numerical", rel < 1e-5, f"max rel. diff {rel:.1e}")

    # 2. CO2 solid branch: 1 atm at 194.7 K (sublimation point), close to the liquid fit at the triple point
    c = Condensible("CO2")
    p = c.p_sat(np.array([194.7]))[0]
    report("CO2 ice: p_sat(194.7 K) = 1.013 bar", abs(p / 1.01325e6 - 1) < 0.03, f"{p / 1e6:.4f} bar")
    lo, hi = c.p_sat(np.array([216.58 - 1e-6, 216.58 + 1e-6]))
    report("CO2 solid/liquid continuity at the triple point", abs(lo / hi - 1) < 0.08,
           f"{lo / 1e6:.3f} vs {hi / 1e6:.3f} bar (5.18 bar measured)")
    Ls, Ll = c.latent_heat(np.array([216.0, 217.0])) * c.M / J_TO_ERG / 1e3
    report("CO2 latent heat sublimation > vaporisation", Ls > Ll, f"{Ls:.1f} vs {Ll:.1f} kJ/mol (lit. ~26 / ~16)")

    # 3. latent heats
    w = Condensible("H2O")
    L373 = w.latent_heat(np.array([373.15]))[0] * w.M / J_TO_ERG / 1e3
    report("H2O L(373 K) ~ 40.7 kJ/mol", abs(L373 / 40.66 - 1) < 0.03, f"{L373:.2f} kJ/mol")
    L250 = w.latent_heat(np.array([250.0]))[0] * w.M / J_TO_ERG / 1e3
    report("H2O ice L(250 K) ~ 51 kJ/mol", abs(L250 / 51.0 - 1) < 0.03, f"{L250:.2f} kJ/mol")
    report("no condensation above T_crit", np.isinf(w.p_sat(np.array([700.0]))[0]))
    Tc = np.array([1.0, 10.0, 50.0])
    report("finite p_sat / xi far below the fits", np.all(np.isfinite(w.ln_psat_xi(Tc)[0])) and
           np.all(np.isfinite(Condensible("N2").ln_psat_xi(Tc)[1])))

    # 4. Leconte+2017 Table 1: H2O in H2 (solar-like H2/He) at T_ref = 300 K: varpi 0.87, M L/RT 16, q_cri 0.070.
    #    Their M L / R T = 16 corresponds to L ~ 40 kJ/mol; L(300 K) is 43.99 kJ/mol (M L / R T = 17.6), so q_cri
    #    comes out ~8% lower than their 0.070.
    M_a = 2.3
    varpi = (w.M - M_a) / w.M
    MLRT = w.M * w.latent_heat(np.array([300.0]))[0] / (R * 300.0)
    qc = q_critical(w, np.array([300.0]), M_a)[0]
    report("L17 Table 1 H2O: varpi", abs(varpi - 0.87) < 0.01, f"{varpi:.3f}")
    report("H2O: M L / R T at 300 K = 17.6 (L = 43.99 kJ/mol; L17 Table 1: 16)", abs(MLRT - 17.64) < 0.2, f"{MLRT:.2f}")
    report("L17 Table 1 H2O: q_cri ~ 0.070", abs(qc / 0.070 - 1) < 0.1, f"{qc:.4f}")

    # 5. pseudoadiabat limits
    thermo = MixtureThermo(["H2", "H2O", "NH3"], [Condensible("H2O"), Condensible("NH3")])
    T = np.array([300.0])
    cp_s = thermo.cp_species(T)[0]
    for xc in (1e-8, 0.01, 0.3, 0.9):
        x = np.array([[1 - xc, xc, 0.0]])
        cp = thermo.cp_mix(x, cp_s[None, :])
        xi = np.array([[w.xi(T)[0], 0.0]])
        x_c = x[:, 1:]
        g = moist_gradient(x_c, xi, np.array([[True, False]]), cp)[0]
        Mc, Ma = w.M, thermo.M[0]
        mu = (1 - xc) * Ma + xc * Mc
        # Leconte+2017 eq. 12 in mass units: (R/(mu cp_m)) (1 + q_c/(1-q_c) M_a L/(R T)) / (1 + q_c/(1-q_c) L/(cp_m T) gamma_s)
        q = xc * Mc / mu
        cp_m = cp[0] / mu
        L = w.latent_heat(T)[0]
        gamma_s = (1 - (Mc - Ma) / Mc * q) * Mc * L / (R * T[0])
        ref = R / (mu * cp_m) * (1 + q / (1 - q) * Ma * L / (R * T[0])) / (1 + q / (1 - q) * L / (cp_m * T[0]) * gamma_s)
        report(f"nabla* = Leconte+2017 eq. 12 (x_H2O = {xc:g})", abs(g / min(ref, R / cp[0]) - 1) < 1e-10, f"{g:.6f} vs {ref:.6f}")
        # Ding & Pierrehumbert 2016 eq. 12: dln p / dln T = (p_sat/p) L/(R_c T) + (p_a/p)(c_pa/R_a)(1 + (c_pc/c_pa + (L/(R_c T) - 1) L/(c_pa T)) r) / (1 + L/(R_a T) r)
        cpa = cp_s[0] / Ma
        cpc = cp_s[1] / Mc
        Ra, Rc = R / Ma, R / Mc
        r_sat = xc * Mc / ((1 - xc) * Ma)
        dlnp = xc * L / (Rc * T[0]) + (1 - xc) * cpa / Ra * (1 + (cpc / cpa + (L / (Rc * T[0]) - 1) * L / (cpa * T[0])) * r_sat) \
            / (1 + L / (Ra * T[0]) * r_sat)
        report(f"nabla* = 1 / Ding&Pierrehumbert2016 eq. 12 (x_H2O = {xc:g})", abs(g * dlnp - 1) < 1e-10, f"{g:.6f} vs {1 / dlnp:.6f}")
    x = np.array([[0.0, 1.0, 0.0]])
    g = moist_gradient(x[:, 1:], np.array([[w.xi(T)[0], 0.0]]), np.array([[True, False]]), thermo.cp_mix(x, cp_s[None, :]))[0]
    report("pure condensible: nabla* = 1/xi (Clausius-Clapeyron)", abs(g * w.xi(T)[0] - 1) < 1e-12, f"{g:.6f}")
    x = np.array([[0.7, 0.2, 0.1]])
    xi2 = np.array([[w.xi(T)[0], Condensible("NH3").xi(T)[0]]])
    cp = thermo.cp_mix(x, cp_s[None, :])
    g_sum = moist_gradient(x[:, 1:], xi2, np.array([[True, False]]), cp)[0]
    g_one = moist_gradient(x[:, 1:2], xi2[:, :1], np.array([[True]]), cp)[0]
    report("multi 'sum' with one saturated species = single species", abs(g_sum / g_one - 1) < 1e-14)
    x = np.array([[0.985, 0.01, 0.005]])
    cp = thermo.cp_mix(x, cp_s[None, :])
    g_one = moist_gradient(x[:, 1:], xi2, np.array([[True, False]]), cp)[0]
    g_two = moist_gradient(x[:, 1:], xi2, np.array([[True, True]]), cp)[0]
    report("dilute: a second saturated species lowers the gradient", g_two < g_one, f"{g_two:.5f} < {g_one:.5f}")

    # 6. numerical pseudoadiabatic ascent vs. nabla*: entropy budget per mole of the species that do not condense,
    #    ds_a + sum_k r_c,k ds_c,k + sum_k (L_k / T) dr_c,k = 0 (r_c,k = n_c,k / n_a, condensate removed;
    #    ideal-gas molar entropies s_i = int c_p,i dT/T - R ln p_i)
    from scipy.optimize import brentq
    cases = [("N2", [("H2O", 0.05)], 320.0), ("N2", [("H2O", 0.4)], 360.0),
             ("H2", [("NH3", 0.05), ("H2O", None)], 290.0), ("H2", [("NH3", 0.3), ("H2O", None)], 330.0)]
    for bg, conds, T0 in cases:
        cs = [Condensible(n) for n, _ in conds]
        th = MixtureThermo([bg] + [c.name for c in cs], cs)
        # pressure from the first condensible's mole fraction; the others saturated at that p
        p0 = cs[0].p_sat(np.array([T0]))[0] / conds[0][1]

        def fractions(T, p):
            return np.array([c.p_sat(np.array([T]))[0] / p for c in cs])

        def entropy_change(T1, p1, T2, p2):
            Tm = 0.5 * (T1 + T2)
            cps = th.cp_species(np.array([Tm]))[0]
            x1, x2 = fractions(T1, p1), fractions(T2, p2)
            xa1, xa2 = 1 - x1.sum(), 1 - x2.sum()
            r1, r2 = x1 / xa1, x2 / xa2
            ds = cps[0] * np.log(T2 / T1) - R * np.log(p2 * xa2 / (p1 * xa1))
            for k, c in enumerate(cs):
                ds_k = cps[1 + k] * np.log(T2 / T1) - R * np.log(p2 * x2[k] / (p1 * x1[k]))
                Lm = c.latent_heat(np.array([Tm]))[0] * c.M
                ds += 0.5 * (r1[k] + r2[k]) * ds_k + (r2[k] - r1[k]) * Lm / Tm
            return ds

        lnp_step = -1e-5
        p1 = p0 * np.exp(lnp_step)
        T1 = brentq(lambda T: entropy_change(T0, p0, T, p1), 0.9 * T0, T0)
        num = np.log(T1 / T0) / lnp_step
        xc = fractions(T0, p0)
        x = np.concatenate(([1 - xc.sum()], xc))[None, :]
        x_sat, xi = th.saturation(np.array([T0]), np.array([p0]))
        g = moist_gradient(x[:, 1:], xi, np.ones_like(xi, bool), th.cp_mix(x, th.cp_species(np.array([T0]))))[0]
        label = "+".join(f"{c.name}({xck:.2g})" for c, xck in zip(cs, xc))
        report(f"nabla* = numerical pseudoadiabatic ascent ({label} in {bg}, {T0:g} K)", abs(num / g - 1) < 2e-3,
               f"{g:.5f} vs {num:.5f}")

    # 7. inhibition parameter: one species = varpi q xi; H2O in N2 negative
    mu = np.array([0.9 * 2.3 + 0.1 * w.M])
    gam = inhibition_gamma(np.array([[0.1]]), np.array([[16.0]]), np.array([[True]]), np.array([w.M]), mu)[0]
    q = 0.1 * w.M / mu[0]
    ref = (w.M - 2.3) / w.M * q * 16.0
    report("Gamma = varpi q xi (one species)", abs(gam / ref - 1) < 1e-12, f"{gam:.4f}")
    mu = np.array([0.9 * 28.0 + 0.1 * w.M])
    report("Gamma < 0 for H2O in N2", inhibition_gamma(np.array([[0.1]]), np.array([[16.0]]), np.array([[True]]),
                                                       np.array([w.M]), mu)[0] < 0)
    print("thermo check:", "all passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="test the formulas")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if _check() else 1)
    ap.print_help()
