"""Innes et al. (2023) reproduction (grids/innes2023/innes2023_{1,10}bar.yaml): summary table + figures.

    python3 grids/innes2023/plot_innes2023.py [--out analyze/images/innes2023]

Per run (T_s = BOA temperature, i.e. HELIOS' T_lay[nlayer] at p_s; T_layer0 = lowest layer): S, convergence, iterations,
p_s, T_s, q(T_s, p_s) (saturated specific humidity at the BOA), q_cri(T_s) (Leconte+2017), regime (1: q < q_cri, 2: q_cri < q < 1, 3: q = 1, i.e.
p_s = p_sat(T_s)), inhibited / saturated layers, Bond albedo (upward TOA flux at lambda < 4 um / incoming), OLR.
Writes <out>_summary.csv, <out>_Ts.png (Fig. 4 analogue) and <out>_profiles.png (Fig. 3 analogue).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from chelio_sim import thermo  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cases(prefix):
    """case label -> output dir; prefix 'innes2023_' (diffusivity 2) or 'innes2023_d1.66_'."""
    return {f"{p} bar{' ' + v if v else ''}": f"output/{prefix}{p}bar{'_' + v if v else ''}"
            for v in ("", "radiative", "noinhib") for p in (1, 10)}
# Innes et al. (2023) Table 1, G star, with moisture: runaway limits [W/m^2]
PAPER_LIMIT = {"1 bar": 530.0, "10 bar": 92.0}
M_H2O = 18.01528
# Innes et al. (2023) Fig. 4, G star, T_s [K] vs S [W/m^2], read off by eye (approximate, for orientation only)
PAPER_FIG4 = {"1 bar": [(232, 248), (300, 257), (350, 262), (400, 266), (450, 271), (470, 277), (480, 300), (490, 322),
                        (500, 345), (510, 368), (507, 372), (515, 405), (525, 450), (530, 480), (532, 600)],
              "10 bar": [(48, 285), (60, 298), (70, 308), (78, 318), (80, 330), (85, 380), (88, 418), (90, 442),
                         (92, 470), (92, 600)]}


def read_column(run_dir):
    f = glob.glob(os.path.join(run_dir, "*_column.dat"))[0]
    lines = open(f).read().splitlines()
    names = lines[2].split()
    data = np.array([[float(v) for v in l.split()] for l in lines[3:]])
    c = {k: data[:, i] for i, k in enumerate(names)}
    m = re.search(r"surface: p = (\S+) dyn/cm2, T = (\S+) K", lines[1])
    c["p_boa"], c["T_boa"] = float(m.group(1)), float(m.group(2))
    return c


def toa_fluxes(run_dir, name):
    """(F_down, F_up, F_up(lambda < 4 um)) at TOA [W/m^2]."""
    up = np.loadtxt(os.path.join(run_dir, f"{name}_spec_upflux.dat"), skiprows=3)
    lam, dlam = up[:, 1], up[:, 3] * 1e-4          # um, cm
    F_up_toa = up[:, -1] * dlam * 1e-3              # erg/s/cm^2 -> W/m^2
    dn = np.loadtxt(os.path.join(run_dir, f"{name}_spec_downflux.dat"), skiprows=3)
    F_dn_toa = dn[:, -1] * dlam * 1e-3
    sw = lam < 4.0
    return F_dn_toa.sum(), F_up_toa.sum(), F_up_toa[sw].sum(), F_up_toa[~sw].sum()


def summarize(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    js = json.load(open(os.path.join(run_dir, f"{name}_session.json")))
    c = read_column(run_dir)
    x = c["x_H2O"]
    mu = c["mu[g/mol]"]
    q = x * M_H2O / mu
    M_a = np.where(x < 1, (mu - x * M_H2O) / np.maximum(1 - x, 1e-30), 2.3)
    q_cri = thermo.q_critical(thermo.Condensible("H2O"), c["T[K]"], M_a)
    S = float(re.search(r"S=([\d.]+)", name).group(1))
    T_s, p_s = c["T_boa"], c["p_boa"]
    # BOA: saturated over the ocean, x = p_sat(T_s) / p_s (capped by the model's base VMR 0.9999)
    p_sat_s = float(thermo.Condensible("H2O").p_sat(np.array([T_s]))[0])
    x_s = min(p_sat_s / p_s, 0.9999)
    M_a_s = float(M_a[0])
    q_s = x_s * M_H2O / (x_s * M_H2O + (1 - x_s) * M_a_s)
    q_cri_s = float(thermo.q_critical(thermo.Condensible("H2O"), np.array([T_s]), M_a_s)[0])
    regime = 3 if p_sat_s >= 0.999 * p_s else (2 if q_s > q_cri_s else 1)
    try:
        F_dn, F_up, F_sw, F_lw = toa_fluxes(run_dir, name)
    except OSError:
        F_dn = F_up = F_sw = F_lw = np.nan
    return dict(S=S, converged=js.get("converged"), rad_iter=js.get("rad_iter"), conv_iter=js.get("conv_iter"),
                p_s_bar=p_s * 1e-6, T_s=T_s, T_layer0=float(c["T[K]"][0]), q_boa=q_s, q_cri_boa=q_cri_s, regime=regime,
                n_inhibited=int(np.sum(c["inhibited"] > 0)), n_gamma_gt1=int(np.sum((c["Gamma"] > 1) & (c["sat_H2O"] > 0))),
                n_conv=int(np.sum(c["conv"] > 0)), albedo=F_sw / F_dn if F_dn else np.nan, OLR=F_lw, F_in=F_dn,
                _c=c, _q=q, _q_cri=q_cri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusivity", default="1.66", choices=["2", "1.66"],
                    help="which set of runs: HELIOS default 2 (output/innes2023_*) or 1.66 (output/innes2023_d1.66_*)")
    ap.add_argument("--out", default=None, help="default analyze/images/innes2023[_d1.66]")
    args = ap.parse_args()
    prefix = "innes2023_" if args.diffusivity == "2" else "innes2023_d1.66_"
    args.out = args.out or os.path.join(ROOT, "analyze", "images", prefix.rstrip("_"))
    CASES = cases(prefix)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {}
    for case, d in CASES.items():
        runs = [r for r in glob.glob(os.path.join(ROOT, d, "S=*")) if glob.glob(os.path.join(r, "*_column.dat"))]
        res[case] = sorted((summarize(r) for r in runs), key=lambda r: r["S"])

    keys = ["S", "converged", "rad_iter", "conv_iter", "p_s_bar", "T_s", "T_layer0", "q_boa", "q_cri_boa", "regime",
            "n_inhibited", "n_gamma_gt1", "n_conv", "albedo", "OLR", "F_in"]
    with open(args.out + "_summary.csv", "w") as f:
        f.write("case," + ",".join(keys) + "\n")
        for case, rows in res.items():
            for r in rows:
                f.write(case + "," + ",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    for case, rows in res.items():
        print(f"\n{case}")
        print(" ".join(f"{k:>10s}" for k in keys))
        for r in rows:
            print(" ".join(f"{r[k]:10.4g}" if isinstance(r[k], float) else f"{str(r[k]):>10s}" for k in keys))

    # T_s vs S per surface pressure: jump / radiative / no inhibition, paper's G-star curve (read off Fig. 4 by eye)
    styles = {"": ("jump (default)", "o", "-"), " radiative": ("radiative", "s", "--"), " noinhib": ("no inhibition", "^", ":")}
    colors = {1: "tab:green", 2: "tab:orange", 3: "tab:pink"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pres in zip(axs, ("1 bar", "10 bar")):
        S_p, T_p = np.array(PAPER_FIG4[pres]).T
        ax.plot(S_p, T_p, "-", color="tab:blue", lw=3, alpha=0.35, label="Innes+23 G star (Fig. 4, by eye)")
        for suffix, (lab, mk, ls) in styles.items():
            rows = res.get(pres + suffix) or []
            if not rows:
                continue
            ax.plot([r["S"] for r in rows], [r["T_s"] for r in rows], ls, color="0.4", lw=1, label=lab, marker=mk,
                    mfc="none", mec="0.4", ms=4)
            for r in rows:
                ax.plot(r["S"], r["T_s"], mk if r["converged"] else "x", color=colors[r["regime"]], ms=7)
        ax.axvline(PAPER_LIMIT[pres], ls="--", color="k", lw=1)
        ax.set_title(f"{pres} H$_2$-He, G star, diffusivity {args.diffusivity} (Innes+23 runaway: {PAPER_LIMIT[pres]:g} W m$^{{-2}}$)",
                     fontsize=9)
        ax.set_xlabel("instellation S [W m$^{-2}$]")
        ax.set_ylabel("T$_s$ (BOA) [K]")
        ax.legend(fontsize=8, loc="upper left")
    for k, lab in [(1, "q < q$_c$"), (2, "q$_c$ < q < 1"), (3, "q = 1 (steam at BOA)")]:
        axs[1].plot([], [], "o", color=colors[k], label=lab)
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out + "_Ts.png", dpi=150)

    # profiles: jump and radiative mode for both pressures
    prof_cases = [c for c in ("1 bar", "1 bar radiative", "10 bar", "10 bar radiative") if res.get(c)]
    fig, axs = plt.subplots(len(prof_cases), 2, figsize=(11, 4.2 * len(prof_cases)), squeeze=False)
    for j, case in enumerate(prof_cases):
        rows = res[case]
        cmap = plt.get_cmap("viridis", max(len(rows), 1))
        for i, r in enumerate(rows):
            c = r["_c"]
            p = np.append(c["p_boa"], c["p[dyn/cm2]"]) * 1e-6
            T = np.append(c["T_boa"], c["T[K]"])
            col = cmap(i)
            axs[j, 0].plot(T, p, color=col, label=f"S={r['S']:g}" + ("" if r["converged"] else " (n.c.)"))
            inh = (c["Gamma"] > 1) & (c["sat_H2O"] > 0)
            axs[j, 0].plot(c["T[K]"][inh], c["p[dyn/cm2]"][inh] * 1e-6, "x", color=col, ms=4)
            axs[j, 1].plot(r["_q"], c["p[dyn/cm2]"] * 1e-6, color=col)
            axs[j, 1].plot(r["_q_cri"], c["p[dyn/cm2]"] * 1e-6, ":", color=col, lw=0.8)
        axs[j, 0].set_ylabel(f"p [bar] ({case})")
        axs[j, 0].legend(fontsize=6, ncol=2)
        for ax in axs[j]:
            ax.set_yscale("log")
            ax.set_ylim(p.max() * 1.2, 1e-6)
        axs[j, 1].set_xscale("log")
        axs[j, 1].set_xlim(1e-12, 2)
    axs[-1, 0].set_xlabel("T [K]  (x: saturated with $\\Gamma$ > 1)")
    axs[-1, 1].set_xlabel("specific humidity q  (dotted: q$_{cri}$)")
    fig.tight_layout()
    fig.savefig(args.out + "_profiles.png", dpi=150)
    print("\nwrote", args.out + "_{summary.csv,Ts.png,profiles.png}")


if __name__ == "__main__":
    main()
