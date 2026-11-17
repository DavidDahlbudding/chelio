"""
Convective adjustment for the session runner's column model: drop-in replacements for HELIOS'
host_functions.conv_check / mark_convective_layers / conv_correct / convective_adjustment (fork commit 3ae6031),
installed by helios_session when a run has a column model (quant.chelio_column).

HELIOS tests and adjusts with half-layer power laws: going up through layer i,
    T(p_int[i]) -> T_lay[i]:      exponent kappa_int[i]   (lower half)
    T_lay[i]   -> T(p_int[i+1]):  exponent kappa_lay[i]   (upper half)
and uses the same exponents as instability threshold and as adjustment target. Here they are separate:
    quant.kappa_lay / kappa_int     TARGET lapse rate (dry or moist adiabat; set by the column model)
    column.thr_lay / thr_int        onset THRESHOLD per pair (i, i+1): its upper half of layer i and lower half of
                                    layer i+1 (and kappa_int[0] for the surface pair) belong to that pair only.
                                    Schwarzschild: = target; Ledoux: target + nabla_mu; saturated pairs: = target
                                    (moist criterion of Leconte et al. 2017); inhibited pairs: +inf (stable).
                                    Always >= target, so an adjusted zone tests stable and the adjustment ends.
conv_check (onset, factor 1+1e-6) uses the threshold; mark_convective_layers (factor 1-1e-6) uses the target for
pairs whose two layers were convective when it was called (mixing has removed nabla_mu: semiconvection
"onset_only") and the threshold otherwise ("strict": always the threshold). With threshold == target both
reproduce HELIOS' flags exactly; conv_correct reproduces HELIOS' temperatures to rounding (O(Z) instead of O(Z^2)).

conv_correct differs from HELIOS in one point: the interface for the flux-balance "fudge factor" of a convective
zone is searched in the first radiative gap above it that is thicker than one pressure scale height (HELIOS) or
that contains an inhibited layer (here): inhibited radiative layers are never stitched, so they must count as real
gaps (else the zone below is controlled by a far-away interface and nothing balances the flux at the inhibited
layer). convective_adjustment caps its loop (HELIOS has no cap).
"""
import logging
from dataclasses import dataclass, fields

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ConvectionSettings:
    """
    criterion        schwarzschild | ledoux (unsaturated pairs: threshold target + max(nabla_mu, 0))
    moist            moist pseudoadiabat as target where saturated (else the dry adiabat of the actual composition)
    inhibition       EXPERIMENTAL. Leconte et al. (2017): moist convection is inhibited where saturated with
                     Gamma > 1. Converges in the test case (grids/column_tests/c1) but is not validated (jump size
                     and position vs. a resolved radiative layer, energy transport through the jump).
    inhibition_mode  jump (default): the inhibited radiative layer is sub-grid (Leconte et al.: with nabla_rad >>
                     nabla_ad it is thin and acts as an almost isobaric temperature jump fixed by thermodynamics):
                     a saturated parcel with Gamma > 1 relaxes to Gamma = 1 over jump_width in ln p (extra lapse rate
                     (Gamma - 1) / (jump_width dGamma/dlnT)), so the targets contain the jump, smoothed over about
                     half a layer, and the zones below and above form one convective zone.
                     radiative: saturated layers with Gamma > 1 are stable, radiative and never stitched (resolved
                     radiative layer; needs the layer to stay inside the narrow temperature band where
                     q_cri < q_sat < q_deep, which a 0.1 dex grid usually cannot)
    semiconvection   onset_only (Ledoux for onset, target for pairs that are already convective) | strict
    multi            sum | dominant (pseudoadiabat with all saturated condensibles, or only the dominant one)
    thermo_source    host (c_p, kappa from the column model) | table (HELIOS' kappa table, dry only; regression)
    inhibition_hysteresis   inhibited if Gamma > 1 + h, released if Gamma < 1 - h
    moist_tolerance  a convective pair with a moist target stays convective unless its upper layer is more than this
                     fraction (relative T) warmer than the target allows (HELIOS: ~1e-6 in the exponent, kept for dry
                     pairs). Moist targets depend strongly on T (near the lifting condensation level the half-layer
                     exponent mixes dry and moist parts and moves with the LCL), so the flux-balance shift of a zone's
                     temperature changes them by far more than 1e-6 and moist zones would fall apart.
    unfreeze         a layer that has been radiative for `adapt_interval` iterations with HELIOS' adaptive time-step
                     prefactor below 1e-5 (it decays while the layer is convective, so the layer is frozen) gets the
                     prefactor 1e-3 (as HELIOS does for all layers at iteration 6000)
    max_adjust_iter  cap of the instability loop in convective_adjustment
    parcel_substeps  RK2 substeps per half layer for the target lapse rates
    """
    criterion: str = "schwarzschild"
    moist: bool = True
    inhibition: bool = True
    inhibition_mode: str = "jump"
    jump_width: float = 0.1
    semiconvection: str = "onset_only"
    multi: str = "sum"
    thermo_source: str = "host"
    inhibition_hysteresis: float = 0.02
    moist_tolerance: float = 1e-3
    unfreeze: bool = True
    max_adjust_iter: int = 1000
    parcel_substeps: int = 2

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown convection settings {sorted(unknown)} (known: {sorted(known)})")
        s = cls(**d)
        s.validate()
        return s

    def validate(self):
        choices = {"criterion": ("schwarzschild", "ledoux"), "semiconvection": ("onset_only", "strict"),
                   "inhibition_mode": ("jump", "radiative"),
                   "multi": ("sum", "dominant"), "thermo_source": ("host", "table")}
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"convection.{key} must be one of {allowed}, got {getattr(self, key)!r}")
        if self.inhibition_hysteresis < 0 or self.inhibition_hysteresis >= 1:
            raise ValueError("convection.inhibition_hysteresis must be in [0, 1)")
        if not 0 < self.jump_width <= 2:
            raise ValueError("convection.jump_width must be in (0, 2] (ln p)")
        if not 0 <= self.moist_tolerance < 0.1:
            raise ValueError("convection.moist_tolerance must be in [0, 0.1)")
        if int(self.parcel_substeps) < 1 or int(self.max_adjust_iter) < 1:
            raise ValueError("convection.parcel_substeps and max_adjust_iter must be >= 1")

    def label(self):
        inh = f"{int(self.inhibition)}" + (f"({self.inhibition_mode})" if self.inhibition else "")
        return (f"{self.criterion},moist={int(self.moist)},inhibition={inh},"
                f"semiconv={self.semiconvection},multi={self.multi},thermo={self.thermo_source}")


# --- HELIOS-equivalent functions with separate threshold and target ---

def n_pairs(quant):
    """Number of layer pairs (i, i+1) HELIOS tests: up to the first layer with p <= 10 dyn/cm^2."""
    n = quant.nlayer
    top = np.flatnonzero(np.asarray(quant.p_lay[:n - 1]) <= 1e1)
    return int(top[0]) if top.size else n - 1


def _pair_unstable(quant, thr_up, thr_lo, factor, slack=1.0):
    """T_lay[i+1] colder than slack x the power-law extrapolation from T_lay[i] with exponents thr_up (upper half of i),
    thr_lo (lower half of i+1), for the pairs i = 0..len-1 (HELIOS' arithmetic for slack = 1)."""
    m = len(thr_up)
    i = np.arange(m)
    T = quant.T_lay
    p_lay, p_int = quant.p_lay, quant.p_int
    with np.errstate(over="ignore", invalid="ignore"):
        T_mid = T[i] * (p_int[i + 1] / p_lay[i]) ** (thr_up * factor)
        T_lim = T_mid * (p_lay[i + 1] / p_int[i + 1]) ** (thr_lo * factor)
    return T[i + 1] < T_lim * slack


def _surface_unstable(quant, thr, factor, slack=1.0):
    n = quant.nlayer
    with np.errstate(over="ignore", invalid="ignore"):
        T_lim = quant.T_lay[n] * (quant.p_lay[0] / quant.p_int[0]) ** (thr * factor)
    return quant.T_lay[0] < T_lim * slack


def conv_check(quant, col):
    """Onset test (HELIOS conv_check with the column thresholds)."""
    n = quant.nlayer
    quant.conv_unstable = np.zeros(n + 1, np.int32)
    m = n_pairs(quant)
    up = _pair_unstable(quant, col.thr_lay[:m], col.thr_int[1:m + 1], 1 + 1e-6)
    idx = np.flatnonzero(up)
    quant.conv_unstable[idx] = 1
    quant.conv_unstable[idx + 1] = 1
    if _surface_unstable(quant, col.thr_int[0], 1 + 1e-6):
        quant.conv_unstable[n] = 1
        quant.conv_unstable[0] = 1


def mark_convective_layers(quant, col):
    """HELIOS mark_convective_layers without stitching (done by the caller), with the semiconvection rule."""
    n = quant.nlayer
    conv = quant.conv_layer
    prev = np.array(conv, int)
    conv[n] = 0
    conv[0] = 0
    m = n_pairs(quant)
    thr_up, thr_lo = col.thr_lay[:m], col.thr_int[1:m + 1]
    thr_s = col.thr_int[0]
    if col.settings.semiconvection == "onset_only":
        mixed = (prev[:m] == 1) & (prev[1:m + 1] == 1)
        thr_up = np.where(mixed & np.isfinite(thr_up), quant.kappa_lay[:m], thr_up)
        thr_lo = np.where(mixed & np.isfinite(thr_lo), quant.kappa_int[1:m + 1], thr_lo)
        if prev[n] == 1 and prev[0] == 1 and np.isfinite(thr_s):
            thr_s = quant.kappa_int[0]
    # moist pairs that are convective stay so unless more than moist_tolerance (relative T) subadiabatic
    factor = np.full(m, 1 - 1e-6)
    slack = np.ones(m)
    tol = col.settings.moist_tolerance
    sat_pair = getattr(col, "sat_pair", None)
    if sat_pair is not None and tol > 0:
        kept = (prev[:m] == 1) & (prev[1:m + 1] == 1) & sat_pair[:m] & np.isfinite(thr_up) & np.isfinite(thr_lo)
        factor[kept] = 1.0
        slack[kept] = 1 + tol
    up = _pair_unstable(quant, thr_up, thr_lo, factor, slack)
    # HELIOS' sequential loop: conv[i] = conv[i+1] = 1 if unstable, else conv[i+1] = 0
    new = np.zeros(m + 1, bool)
    new[:m] |= up
    new[1:] |= up
    conv[:m + 1] = new.astype(conv.dtype)
    # no kinks at the top edge of a convective zone
    T = quant.T_lay
    kink = np.flatnonzero(T[1:n] > T[:n - 1])
    conv[kink] = 0
    kept_s = prev[n] == 1 and prev[0] == 1 and getattr(col, "sat_surf", False) and tol > 0 and np.isfinite(thr_s)
    if _surface_unstable(quant, thr_s, 1.0 if kept_s else 1 - 1e-6, 1 + tol if kept_s else 1.0):
        conv[n] = 1
        conv[0] = 1


def _zones(quant):
    """Start/end layers of the zones to adjust (HELIOS' convention: -1 = surface ghost layer)."""
    n = quant.nlayer
    flags = (np.asarray(quant.conv_unstable[:n + 1]) == 1) | (np.asarray(quant.conv_layer[:n + 1]) == 1)
    lst = [int(i) for i in np.flatnonzero(flags)]
    if n in lst:
        lst = [-1] + lst[:-1]
    members = set(lst)
    starts = [i for i in lst if i - 1 not in members]
    ends = [i for i in lst if i + 1 not in members]
    return starts, ends


def conv_correct(quant, col, fudging):
    """HELIOS conv_correct (enthalpy-conserving power-law adiabats + fudge factor), O(Z); inhibited gaps are thick."""
    n = quant.nlayer
    starts, ends = _zones(quant)
    if len(starts) != len(ends):
        raise RuntimeError("Error in convective calculation (zone starts/ends)")
    inhibited = getattr(col, "inhibited", None)
    p_lay, p_int = quant.p_lay, quant.p_int
    fudge = np.ones(len(starts))

    if fudging == 1:
        for z in range(len(starts)):
            itt = None
            for m in range(z, len(starts)):
                if m != len(starts) - 1:
                    p_top = p_lay[starts[m + 1]]
                    p_bot = p_lay[ends[m]] if ends[m] != -1 else p_int[0]
                    thick = (p_top / p_bot) < (1 / np.e)
                    if not thick and inhibited is not None:
                        gap = np.arange(ends[m] + 1, starts[m + 1])
                        thick = bool(gap.size and np.any(inhibited[gap]))
                    if thick:
                        itt = int((ends[m] + starts[m + 1]) / 2)
                        break
                else:
                    itt = int(0.8 * ends[m] + 0.2 * (quant.ninterface - 1))
            if quant.input_dampara == "automatic":
                if quant.T_star > 10:
                    quant.dampara = 0.5 if z < len(starts) - 1 else 4.0
                else:
                    quant.dampara = 8.0
            else:
                quant.dampara = float(quant.input_dampara)
            f = ((quant.F_intern + quant.F_add_heat_sum[itt - 1] + quant.F_smooth_sum[itt - 1] + quant.F_down_tot[itt])
                 / quant.F_up_tot[itt]) ** (1.0 / quant.dampara)
            fudge[z] = min(1.01, max(0.99, f))
        col.last_fudge = [[int(starts[z]), int(ends[z]), float(fudge[z])] for z in range(len(starts))]

    k_lay, k_int = quant.kappa_lay, quant.kappa_int
    ln_lower = np.log(p_lay / p_int[:n])          # lower half: p_int[i] -> p_lay[i]
    ln_upper = np.log(p_int[1:n + 1] / p_lay)     # upper half: p_lay[i] -> p_int[i+1]
    for z in range(len(starts)):
        s, e = max(0, starts[z]), max(0, ends[z])
        idx = np.arange(s, e + 1)
        full = k_int[idx] * ln_lower[idx] + k_lay[idx] * ln_upper[idx]
        logf = np.concatenate(([0.0], np.cumsum(full[:-1]))) + k_int[idx] * ln_lower[idx]
        f = np.exp(logf)
        w = quant.c_p_lay[idx] / quant.meanmolmass_lay[idx] * (p_int[idx] - p_int[idx + 1])
        theta = np.sum(w * quant.T_lay[idx]) / np.sum(w * f) * fudge[z]
        quant.T_lay[idx] = theta * f
        if starts[z] == -1:
            quant.T_lay[n] = theta


def convective_adjustment(quant, col, check, mark, correct):
    """HELIOS convective_adjustment with an iteration cap; check/mark/correct are the (patched) HELIOS functions."""
    check(quant)
    unstable = np.sum(quant.conv_unstable) > 0
    it = 0
    cap = int(col.settings.max_adjust_iter)
    while unstable:
        mark(quant, stitching=0)
        correct(quant, fudging=0)
        check(quant)
        unstable = np.sum(quant.conv_unstable) > 0
        it += 1
        if it >= cap and unstable:
            col.adjust_cap_hits += 1
            if col.adjust_cap_hits == 1:
                bad = np.flatnonzero(quant.conv_unstable)
                log.warning(f"convective_adjustment: still unstable after {cap} passes (layers {bad.tolist()[:10]}), "
                            f"continuing; further hits are counted in the column stats")
            break
    mark(quant, stitching=1)
    correct(quant, fudging=1)
