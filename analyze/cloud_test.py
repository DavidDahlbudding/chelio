"""
Analysis of the first cloud test (grids/cloud_test.yaml): one H2O-ice deck in the representative FFP-exomoon case.

    python3 analyze/cloud_test.py                 # from the Chelio root
    python3 analyze/cloud_test.py --checks        # also print the smoke-test checks for every cloudy run

Writes analyze/images/cloud_test_summary.csv and the figures analyze/images/cloud_test_*.png.
Runs that do not exist (yet) are listed with NaNs and left out of the figures.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyze"))

from analyze_modules import ChelioRun  # noqa: E402
from chelio_sim import clouds, grid  # noqa: E402

SIGMA_SB = 5.670374419e-5  # erg s^-1 cm^-2 K^-4
P_LEVELS_BAR = (1.0, 0.1, 0.01)
REFERENCE_RUN = ROOT / "output" / "EqCond+Remove" / "Earth_P0=1e7_Tint=100_CplusO=1e-2_CtoO=0.59_aN=1e-4"


# --- Reading one run ---

def helios_file(run_dir, name, suffix):
    return Path(run_dir) / f"{name}_{suffix}.dat"


def coupling_status(run_dir, name):
    """(converged, number of coupling iterations, total HELIOS iterations)"""
    conv_file = helios_file(run_dir, name, "coupling_convergence")
    converged = conv_file.exists() and conv_file.read_text().strip() == "1"
    pattern = re.compile(rf"{re.escape(name)}_tp_coupling_(\d+)\.dat")
    iters = [int(m.group(1)) for f in os.listdir(run_dir) if (m := pattern.fullmatch(f))]
    n_coupling = max(iters) + 1 if iters else 0
    info = helios_file(run_dir, name, "run_info")
    helios_iters = sum(int(m.group(1)) for m in re.finditer(r"took (\d+) iterations", info.read_text())) if info.exists() else np.nan
    return converged, n_coupling, helios_iters


def rcb(pressure_bar, convective_flag):
    """RCB pressure [bar] and index of the last convective layer (as analyze_modules.plot_rcb_height)."""
    idx = np.where(np.diff(np.nan_to_num(convective_flag).astype(int)) == -1)[0]
    if idx.size == 0:
        return np.nan, None
    i = idx[0]
    return 10 ** ((np.log10(pressure_bar[i]) + np.log10(pressure_bar[i + 1])) / 2), i


def interp_log_p(p_bar, values, p_target, log_values=False):
    """values at p_target [bar], interpolated linearly in log p (and log values if log_values)."""
    x = np.log10(p_bar[::-1])
    y = np.asarray(values)[::-1]
    if log_values:
        return 10 ** np.interp(np.log10(p_target), x, np.log10(np.maximum(y, 1e-300)))
    return np.interp(np.log10(p_target), x, y)


def cloud_optical_depth(run_dir, name):
    """HELIOS bins (centre, width [micron]) and the total cloud optical depth per bin from _cloud_optdepth.dat."""
    path = helios_file(run_dir, name, "cloud_optdepth")
    if not path.exists():
        return None
    data = np.loadtxt(path, skiprows=2)
    return data[:, 1], data[:, 3], data[:, 4:].sum(axis=1)


def load_run(out_dir, name):
    """Everything the table and figures need from one run, or None if the run has no output yet."""
    run_dir = Path(out_dir) / name
    if not (run_dir / "vertical_mix_0.dat").exists():
        return None
    run = ChelioRun(out_dir, name)
    run.read_data()
    run.convert_to_vmr()
    d = run.get_iteration_data(-1)
    if "error" in d:
        return None
    converged, n_coupling, helios_iters = coupling_status(run_dir, name)
    p, T = d["pressure_bar"], d["temperature_K"]
    p_rcb, i_rcb = rcb(p, d["convective_flag"])
    vmr = {m: d["mols_vmr"][:, run.mol_names.index(m)] for m in ("H2O", "NH3", "CH4") if m in run.mol_names}
    settings_file = run_dir / "cloud_settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else None
    return dict(run_dir=run_dir, p=p, T=T, p_rcb=p_rcb, i_rcb=i_rcb, vmr=vmr, converged=converged,
                n_coupling=n_coupling, helios_iters=helios_iters, settings=settings,
                optdepth=cloud_optical_depth(run_dir, name))


# --- Smoke-test checks (plan step 7) ---

def smoke_checks(name, task, res, T_int):
    """Prints the checks of one run: cloud profile, realised tau, energy balance, NaNs in the log."""
    run_dir = res["run_dir"]
    print(f"\n{name}")
    f_bot = float(task.params.get("cloud_bottom_mixing_ratio", 0))
    if f_bot > 0:
        p_bot = float(task.params["cloud_bottom_pressure"])  # dyn/cm^2
        h = float(task.params["cloud_to_gas_scale_height_ratio"])
        data = np.loadtxt(helios_file(run_dir, name, "cloud_mixing_ratio"), skiprows=2)
        p_lay, f = data[:, 1], data[:, 2]  # 10^-6 bar = dyn/cm^2
        i_bot = int(np.argmax(f > 0))
        p_int_low = np.sqrt(p_lay[i_bot - 1] * p_lay[i_bot]) if i_bot > 0 else np.inf
        p_int_up = np.sqrt(p_lay[i_bot] * p_lay[i_bot + 1])
        above = slice(i_bot + 1, len(f))
        expected = f_bot * (p_lay[above] / p_lay[i_bot]) ** (1 / h - 1)
        print(f"  cloud base layer {i_bot}: p = {p_lay[i_bot]:.4g} dyn/cm^2 (approx. interfaces {p_int_low:.4g}-{p_int_up:.4g}), "
              f"p_bot = {p_bot:.4g}; contains p_bot: {p_int_up < p_bot <= p_int_low}")
        print(f"  f = 0 below the base: {bool(np.all(f[:i_bot] == 0))}; f(base) = {f[i_bot]:.4g} (f_bot {f_bot:.4g}); "
              f"f propto p^(1/h-1) above: max rel. dev. {np.max(np.abs(f[above] / expected - 1)):.1e}")
    if res["optdepth"] is not None:
        lam, dlam, tau = res["optdepth"]
        tau_planck = clouds.planck_mean(lam, tau, T_int, dlam_um=dlam)
        target = float(task.params.get("cloud_tau", 0))
        rel = f" ({100 * (tau_planck / target - 1):+.1f} % vs. target {target:g})" if target else ""
        print(f"  realised cloud tau (Planck mean at {T_int:g} K, HELIOS bins): {tau_planck:.4g}{rel}")
    flux = np.genfromtxt(helios_file(run_dir, name, "integrated_flux"), skip_header=3, usecols=(0, 1, 2, 3, 4))
    F_down, F_up, F_net = flux[-1, 2], flux[-1, 3], flux[-1, 4]
    print(f"  TOA: F_net = {F_net:.6g} erg/s/cm^2 = {F_net / (SIGMA_SB * T_int**4):.5f} sigma T_int^4 "
          f"(F_down {F_down:.4g}, F_up {F_up:.6g}); negative fluxes: {bool(np.any(flux[:, 2:4] < 0))}")
    log = (run_dir / "run.log").read_text(errors="replace") if (run_dir / "run.log").exists() else ""
    nan_lines = [line for line in log.splitlines() if re.search(r"\bnan\b", line, re.IGNORECASE)]
    print(f"  NaN mentions in run.log: {len(nan_lines)}" + (f" (first: {nan_lines[0][:100]!r})" if nan_lines else ""))
    print(f"  converged: {res['converged']} after {res['n_coupling']} coupling iterations ({res['helios_iters']} HELIOS iterations)")


# --- Table ---

def build_table(tasks, results, T_int):
    base = next((results[t.name] for t in tasks if float(t.params.get("cloud_tau", 0)) == 0 and results[t.name]), None)
    T_surf_base = base["T"][0] if base else np.nan
    rows = []
    for task in tasks:
        res = results[task.name]
        row = {
            "p_bot_bar": float(task.params.get("cloud_pbot_bar", np.nan)),
            "tau_target": float(task.params.get("cloud_tau", 0)),
            "f_bot": float(task.params.get("cloud_bottom_mixing_ratio", 0)),
        }
        if res is None:
            rows.append({**row, "name": task.name})
            continue
        if res["optdepth"] is not None:
            lam, dlam, tau = res["optdepth"]
            row["tau_realised"] = clouds.planck_mean(lam, tau, T_int, dlam_um=dlam)
        row.update(converged=res["converged"], n_coupling=res["n_coupling"], helios_iterations=res["helios_iters"],
                   T_surf=res["T"][0], dT_surf=res["T"][0] - T_surf_base, p_rcb_bar=res["p_rcb"])
        for p_level in P_LEVELS_BAR:
            row[f"T_{p_level:g}bar"] = interp_log_p(res["p"], res["T"], p_level)
        for mol in ("H2O", "NH3"):
            if mol in res["vmr"]:
                row[f"{mol}_surf"] = res["vmr"][mol][0]
                row[f"{mol}_rcb"] = interp_log_p(res["p"], res["vmr"][mol], res["p_rcb"], log_values=True) if np.isfinite(res["p_rcb"]) else np.nan
        rows.append({**row, "name": task.name})
    return rows


COLUMNS = ["p_bot_bar", "tau_target", "tau_realised", "f_bot", "converged", "n_coupling", "helios_iterations", "T_surf", "dT_surf",
           "p_rcb_bar", "T_1bar", "T_0.1bar", "T_0.01bar", "H2O_surf", "H2O_rcb", "NH3_surf", "NH3_rcb", "name"]


def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.4g}"


def write_table(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row.get(c, "") if not isinstance(row.get(c), float) else f"{row[c]:.6g}" for c in COLUMNS])
    print(f"\nSaved {path}")


def print_table(rows):
    cols = [c for c in COLUMNS if c != "name"]
    widths = [max(len(c), *(len(fmt(r.get(c, np.nan))) for r in rows)) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt(r.get(c, np.nan)).rjust(w) for c, w in zip(cols, widths)))


# --- Figures ---

def tau_colors(taus):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    taus = sorted(taus)
    return {tau: cmap(0.15 + 0.7 * i / max(len(taus) - 1, 1)) for i, tau in enumerate(taus)}


def plot_tp(tasks, results, reference, path):
    import matplotlib.pyplot as plt
    cloudy = [t for t in tasks if float(t.params.get("cloud_tau", 0)) > 0]
    p_bots = sorted({float(t.params["cloud_pbot_bar"]) for t in cloudy}, reverse=True)
    colors = tau_colors({float(t.params["cloud_tau"]) for t in cloudy})
    base = next((results[t.name] for t in tasks if float(t.params.get("cloud_tau", 0)) == 0), None)
    fig, axes = plt.subplots(1, len(p_bots), figsize=(4.2 * len(p_bots), 5), sharey=True, squeeze=False)
    for ax, p_bot in zip(axes[0], p_bots):
        if reference is not None:
            ax.plot(reference["T"], reference["p"], color="0.6", ls=":", lw=1.5, label="published (Zenodo)")
        if base is not None:
            ax.plot(base["T"], base["p"], color="k", lw=1.5, label="cloud-free")
            ax.plot(np.interp(np.log10(base["p_rcb"]), np.log10(base["p"][::-1]), base["T"][::-1]), base["p_rcb"], "o", color="k", ms=5)
        for task in sorted((t for t in cloudy if float(t.params["cloud_pbot_bar"]) == p_bot), key=lambda t: float(t.params["cloud_tau"])):
            res = results[task.name]
            if res is None:
                continue
            tau = float(task.params["cloud_tau"])
            ls = "-" if res["converged"] else "--"
            ax.plot(res["T"], res["p"], color=colors[tau], ls=ls, label=rf"$\tau$ = {tau:g}" + ("" if res["converged"] else " (not conv.)"))
            if np.isfinite(res["p_rcb"]):
                ax.plot(np.interp(np.log10(res["p_rcb"]), np.log10(res["p"][::-1]), res["T"][::-1]), res["p_rcb"], "o", color=colors[tau], ms=5)
        ax.axhline(p_bot, color="C3", ls="--", lw=1, label="cloud base")
        ax.set_yscale("log")
        ax.set_title(f"cloud base {p_bot:g} bar")
        ax.set_xlabel("T [K]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel("p [bar]")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(10.5, 1e-7)
    fig.suptitle("H$_2$O-ice deck, T$_{int}$ = 100 K, P$_{surf}$ = 10 bar (dots: RCB)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dT(rows, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, p_bot in enumerate(sorted({r["p_bot_bar"] for r in rows if np.isfinite(r["p_bot_bar"])}, reverse=True)):
        sel = sorted((r for r in rows if r["p_bot_bar"] == p_bot and np.isfinite(r.get("dT_surf", np.nan))), key=lambda r: r["tau_target"])
        if sel:
            ax.plot([r["tau_target"] for r in sel], [r["dT_surf"] for r in sel], "o-", color=f"C{i}", label=f"base {p_bot:g} bar")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"cloud optical depth $\tau$ (Planck mean, 100 K)")
    ax.set_ylabel(r"$\Delta T_\mathrm{surf}$ [K]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_vmr(tasks, results, path):
    import matplotlib.pyplot as plt
    cloudy = [t for t in tasks if float(t.params.get("cloud_tau", 0)) > 0]
    p_bots = sorted({float(t.params["cloud_pbot_bar"]) for t in cloudy}, reverse=True)
    colors = tau_colors({float(t.params["cloud_tau"]) for t in cloudy})
    base = next((results[t.name] for t in tasks if float(t.params.get("cloud_tau", 0)) == 0), None)
    mols = ("H2O", "NH3", "CH4")
    fig, axes = plt.subplots(len(p_bots), len(mols), figsize=(4 * len(mols), 3.6 * len(p_bots)), sharey=True, squeeze=False)
    for row, p_bot in enumerate(p_bots):
        for col, mol in enumerate(mols):
            ax = axes[row, col]
            if base is not None and mol in base["vmr"]:
                ax.plot(base["vmr"][mol], base["p"], color="k", lw=1.5, label="cloud-free")
            for task in sorted((t for t in cloudy if float(t.params["cloud_pbot_bar"]) == p_bot), key=lambda t: float(t.params["cloud_tau"])):
                res = results[task.name]
                if res is None or mol not in res["vmr"]:
                    continue
                tau = float(task.params["cloud_tau"])
                ax.plot(res["vmr"][mol], res["p"], color=colors[tau], label=rf"$\tau$ = {tau:g}")
            ax.axhline(p_bot, color="C3", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-12, 1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{mol}, cloud base {p_bot:g} bar", fontsize=10)
            if row == len(p_bots) - 1:
                ax.set_xlabel("VMR")
            if col == 0:
                ax.set_ylabel("p [bar]")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(10.5, 1e-7)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default=str(ROOT / "grids" / "cloud_test.yaml"))
    parser.add_argument("--images", default=str(ROOT / "analyze" / "images"))
    parser.add_argument("--checks", action="store_true", help="print the smoke-test checks for every run")
    args = parser.parse_args()

    spec = grid.load_grid(args.grid)
    tasks = grid.expand_tasks(spec)
    out_dir = ROOT / spec["out_dir"]
    T_int = float(tasks[0].params["internal_temp"])
    results = {t.name: load_run(out_dir, t.name) for t in tasks}
    reference = load_run(REFERENCE_RUN.parent, REFERENCE_RUN.name) if REFERENCE_RUN.exists() else None

    if args.checks:
        for task in tasks:
            if results[task.name] is not None:
                smoke_checks(task.name, task, results[task.name], T_int)
        print()

    rows = build_table(tasks, results, T_int)
    if reference is not None:
        print(f"Published reference ({REFERENCE_RUN.name}): T_surf = {reference['T'][0]:.2f} K, p_RCB = {reference['p_rcb']:.3g} bar, "
              + ", ".join(f"T({p:g} bar) = {interp_log_p(reference['p'], reference['T'], p):.1f} K" for p in P_LEVELS_BAR))
    print_table(rows)

    os.makedirs(args.images, exist_ok=True)
    write_table(rows, os.path.join(args.images, "cloud_test_summary.csv"))
    if any(r is not None for r in results.values()):
        plot_tp(tasks, results, reference, os.path.join(args.images, "cloud_test_TP.png"))
        plot_dT(rows, os.path.join(args.images, "cloud_test_dTsurf.png"))
        plot_vmr(tasks, results, os.path.join(args.images, "cloud_test_vmr.png"))

    # size-averaged ice optics (same figure as make_mie_files --check)
    from chelio_sim.make_mie_files import plot_optics
    params = tasks[0].params
    mie_dir = ROOT / str(params["cloud_mie_path"])
    lam, c_ext, c_sca, g = clouds.size_averaged_optics(mie_dir, float(params["cloud_r_mode"]), float(params["cloud_r_std_dev"]))
    plot_optics(lam, c_ext, c_sca, g, os.path.join(args.images, "cloud_test_ice_optics.png"),
                f"{mie_dir.name}: size-averaged optics, r_mode = {params['cloud_r_mode']:g} micron, sigma_g = {params['cloud_r_std_dev']:g}")


if __name__ == "__main__":
    main()
