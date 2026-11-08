#!/usr/bin/env python3
"""
Compares the T-P profiles, convergence and run times of the session tests (grids/session_tests/*.yaml, slurm/
session_tests.sbatch) with a reference test (default: the legacy coupled runs).

    python analyze/compare_session_tests.py                                   # table + plot of all tests found
    python analyze/compare_session_tests.py --check t2a_grid_cold t2b_grid_warm --threshold 10
        exit code 10 if any run of the checked tests differs from the reference by more than --threshold K
        (the slurm script uses this to decide whether Test 3, the refined chemistry grid, is run)

Final profile of a run: the highest <name>_tp_coupling_<i>.dat (session runs: i = 0). Wall times from
<test dir>/grid_timings.csv (written by run_grid.py), HELIOS iterations from <name>_session.json (session) or
<name>_run_info.dat (coupled; summed over the coupling iterations).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["t1_legacy", "t2a_grid_cold", "t2b_grid_warm", "t3_grid_fine", "t4_grid_init", "t5_fast"]


def final_profile(run_dir):
    name = run_dir.name
    files = [(int(m.group(1)), f) for f in run_dir.glob(f"{name}_tp_coupling_*.dat")
             if (m := re.fullmatch(rf"{re.escape(name)}_tp_coupling_(-?\d+)\.dat", f.name)) and int(m.group(1)) >= 0]
    if not files:
        return None, None
    i, path = max(files)
    data = np.loadtxt(path, skiprows=1)
    return data, i


def run_stats(run_dir):
    name = run_dir.name
    stats = {}
    conv = run_dir / f"{name}_coupling_convergence.dat"
    stats["converged"] = conv.exists() and conv.read_text().strip() == "1"
    info = run_dir / f"{name}_session.json"
    if info.exists():
        d = json.loads(info.read_text())
        stats.update(iterations=(d.get("rad_iter") or 0) + d["conv_iter"], warm=bool(d.get("warm_start")),
                     stitched=len(d.get("stitched_layers_final") or []))
    run_info = run_dir / f"{name}_run_info.dat"
    if run_info.exists():
        its = [int(m.group(1)) for m in re.finditer(r"took (\d+) iterations", run_info.read_text())]
        stats["iterations"] = sum(its)
        stats["coupling_iterations"] = len(its)
    return stats


def timings(test_dir):
    path = test_dir / "grid_timings.csv"
    out = {}
    if path.exists():
        for row in csv.DictReader(path.open()):
            if row["status"] == "done":
                out[row["name"]] = float(row["wall_time_s"])
    return out


def diff(profile, reference):
    """max |dT| and dT at the surface, on the reference pressures (interpolated in log P)."""
    lp, lr = np.log10(profile[:, 0]), np.log10(reference[:, 0])
    order = np.argsort(lp)
    T = np.interp(lr, lp[order], profile[order, 1])
    d = T - reference[:, 1]
    return float(np.max(np.abs(d))), float(d[np.argmax(reference[:, 0])])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="output/session_tests", help="directory with the test output dirs")
    parser.add_argument("--reference", default="t1_legacy")
    parser.add_argument("--tests", nargs="*", default=TESTS)
    parser.add_argument("--check", nargs="*", default=None, help="tests to check against --threshold")
    parser.add_argument("--threshold", type=float, default=10.0, help="K")
    parser.add_argument("--plot", default=None, help="PNG path (default <root>/comparison.png)")
    args = parser.parse_args()

    root = ROOT / args.root
    tests = [t for t in args.tests if (root / t).is_dir()]
    ref_dir = root / args.reference
    names = sorted({p.name for t in tests for p in (root / t).iterdir() if p.is_dir() and p.name != "chem_grids"},
                   key=lambda n: -float(re.sub(r"[^0-9.]", "", n) or 0))

    rows, profiles = [], {}
    for test in tests:
        t_wall = timings(root / test)
        for name in names:
            run_dir = root / test / name
            if not run_dir.is_dir():
                continue
            prof, i = final_profile(run_dir)
            if prof is None:
                continue
            profiles[(test, name)] = prof
            s = run_stats(run_dir)
            ref, _ = final_profile(ref_dir / name) if (ref_dir / name).is_dir() else (None, None)
            dmax, dsurf = diff(prof, ref) if ref is not None and test != args.reference else (np.nan, np.nan)
            rows.append(dict(test=test, run=name, converged=s.get("converged"), T_surf=prof[np.argmax(prof[:, 0]), 1],
                             dT_max=dmax, dT_surf=dsurf, wall_s=t_wall.get(name, np.nan),
                             iterations=s.get("iterations", np.nan), coupling=s.get("coupling_iterations", ""),
                             stitched=s.get("stitched", "")))

    header = ["test", "run", "converged", "T_surf [K]", f"max|dT| vs {args.reference} [K]", "dT_surf [K]",
              "wall [s]", "HELIOS iter.", "coupl. iter.", "stitched"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append(f"| {r['test']} | {r['run']} | {r['converged']} | {r['T_surf']:.1f} | {r['dT_max']:.2f} | {r['dT_surf']:.2f} "
                     f"| {r['wall_s']:.0f} | {r['iterations']} | {r['coupling']} | {r['stitched']} |")
    totals = ["", "| test | total wall time [s] | runs |", "|---|---|---|"]
    for test in tests:
        w = [r["wall_s"] for r in rows if r["test"] == test and np.isfinite(r["wall_s"])]
        totals.append(f"| {test} | {sum(w):.0f} | {len(w)} |")
    text = "\n".join(lines + totals)
    print(text)
    (root / "comparison.md").write_text(text + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        styles = dict(zip(tests, ["-", "--", ":", "-.", (0, (1, 3)), (0, (5, 1))]))
        colors = {n: plt.cm.viridis(k / max(len(names) - 1, 1)) for k, n in enumerate(names)}
        for (test, name), prof in profiles.items():
            ax.plot(prof[:, 1], prof[:, 0], ls=styles[test], color=colors[name], lw=1.2)
        for test in tests:
            ax.plot([], [], ls=styles[test], color="k", label=test)
        for name in names:
            ax.plot([], [], color=colors[name], label=name)
        ax.set(yscale="log", xlabel="T [K]", ylabel="P [bar]")
        ax.invert_yaxis()
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.plot or root / "comparison.png", dpi=150)
    except ImportError:
        pass

    if args.check:
        bad = [r for r in rows if r["test"] in args.check and np.isfinite(r["dT_max"]) and r["dT_max"] > args.threshold]
        if bad:
            print(f"\n{len(bad)} runs of {args.check} differ by more than {args.threshold} K from {args.reference}")
            sys.exit(10)


if __name__ == "__main__":
    main()
