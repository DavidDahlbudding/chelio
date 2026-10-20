"""
Mie tables for HELIOS cloud decks, computed with LX-MIE (pip package `lxmie`).

    python3 -m chelio_sim.make_mie_files \\
        --ri helios_inputs/clouds/refractive_index/H2O_s.dat \\
        --out helios_inputs/clouds/H2O_s/ \\
        --opacity_dir $HELIOS_PATH/input/opacity/r50_kdistr/ \\
        --n_lambda 2000
    # debugging only: --const_nk 1.5 0.01 instead of --ri

    python3 -m chelio_sim.make_mie_files --check --out helios_inputs/clouds/H2O_s/   # validate existing tables

Writes one file per HELIOS radius (see chelio_sim/clouds.py for the file contract) plus meta.json.
The wavelength grid is log-spaced from 0.8 x the smallest to 1.25 x the largest HELIOS interface wavelength,
so every HELIOS bin is covered (HELIOS silently sets the cloud opacity to zero outside the Mie grid).
"""
import argparse
import contextlib
import datetime
import importlib.metadata
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chelio_sim import clouds

CHELIO_ROOT = Path(__file__).resolve().parent.parent
LAMBDA_MARGIN = (0.8, 1.25)  # Mie grid = [0.8 lambda_min, 1.25 lambda_max] of the HELIOS interface wavelengths

# Size-averaged reference values for H2O ice (Warren 1984), r_mode = 2 micron, sigma_g = 2, from the LX-MIE CLI
# on a 400-point grid (0.3-1000 micron) with the HELIOS weighting and the corrected g (cloud test plan)
REFERENCE_PLANCK_C_EXT = {100: 1.12e-6, 150: 1.39e-6}  # T [K] -> cm^2
REFERENCE_AT_LAMBDA = {10: (1.6e-6, 0.65, 0.91), 20: (1.9e-6, 0.72, 0.74), 50: (1.5e-6, 0.36, 0.53), 100: (5.3e-7, 0.54, 0.46)}


def default_opacity_dir():
    helios_path = os.environ.get("HELIOS_PATH")
    return os.path.join(helios_path, "input", "opacity", "r50_kdistr") if helios_path else None


def read_refractive_index(path):
    """Wavelength [micron], n, k from an LX-MIE compilation file ('#' comment lines). Returns the data and the comments."""
    comments = [line.strip().lstrip("#").strip() for line in open(path) if line.startswith("#")]
    data = np.loadtxt(path, comments="#")
    lam, n, k = data[:, 0], data[:, 1], data[:, 2]
    if np.any(np.diff(lam) <= 0):
        raise ValueError(f"{path}: wavelengths are not strictly ascending")
    return lam, n, k, comments


def interpolate_index(lam_ri, n_ri, k_ri, lam):
    """n linear in log(lambda), k log-log. No extrapolation: lam must lie inside lam_ri."""
    if lam[0] < lam_ri[0] or lam[-1] > lam_ri[-1]:
        raise ValueError(
            f"the Mie wavelength grid ({lam[0]:.4g}-{lam[-1]:.4g} micron) is not inside the range of the "
            f"refractive-index file ({lam_ri[0]:.4g}-{lam_ri[-1]:.4g} micron); not extrapolating"
        )
    # only the tabulated points that bracket the grid are used for k
    used = slice(max(np.searchsorted(lam_ri, lam[0], side="right") - 1, 0), np.searchsorted(lam_ri, lam[-1]) + 1)
    if np.any(k_ri[used] <= 0):
        raise ValueError("k must be > 0 in the wavelength range of the Mie grid (it is interpolated in log-log)")
    log_lam, log_lam_ri = np.log(lam), np.log(lam_ri)
    n = np.interp(log_lam, log_lam_ri, n_ri)
    k = np.exp(np.interp(log_lam, log_lam_ri[used], np.log(k_ri[used])))
    return n, k


def git_hash():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CHELIO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=CHELIO_ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_tables(args):
    import lxmie

    lam_lo, lam_hi = clouds.helios_interface_range(args.opacity_dir)
    lam = np.geomspace(LAMBDA_MARGIN[0] * lam_lo, LAMBDA_MARGIN[1] * lam_hi, args.n_lambda)
    print(f"HELIOS interface wavelengths: {lam_lo:.4g}-{lam_hi:.4g} micron ({args.opacity_dir})")
    print(f"Mie grid: {args.n_lambda} log-spaced points, {lam[0]:.4g}-{lam[-1]:.4g} micron")

    if args.const_nk:
        n = np.full_like(lam, args.const_nk[0])
        k = np.full_like(lam, args.const_nk[1])
        index_info = {"constant_n": args.const_nk[0], "constant_k": args.const_nk[1]}
        print(f"Constant refractive index n = {args.const_nk[0]}, k = {args.const_nk[1]} (debugging only)")
    else:
        lam_ri, n_ri, k_ri, comments = read_refractive_index(args.ri)
        n, k = interpolate_index(lam_ri, n_ri, k_ri, lam)
        index_info = {
            "file": os.path.relpath(os.path.abspath(args.ri), CHELIO_ROOT),
            "reference": comments[0] if comments else "",
            "file_wavelength_range_micron": [float(lam_ri[0]), float(lam_ri[-1])],
        }
        print(f"Refractive index: {args.ri} ({index_info['reference']})")

    os.makedirs(args.out, exist_ok=True)
    header = ("#wavelength[micron] size_parameter C_ext[cm^2] C_sca[cm^2] C_abs[cm^2] "
              "single_scattering_albedo asymmetry_parameter")
    for r in clouds.HELIOS_RADII:
        x = 2 * np.pi * r / lam  # r, lam in micron
        res = lxmie.mie(list(n), list(k), list(x))
        area = np.pi * (r * 1e-4) ** 2  # cm^2
        q_ext, q_sca, q_abs, g = (np.array([getattr(o, a) for o in res])
                                  for a in ("q_ext", "q_sca", "q_abs", "asymmetry_parameter"))
        table = np.column_stack([lam, x, q_ext * area, q_sca * area, q_abs * area, q_sca / q_ext, g])
        np.savetxt(os.path.join(args.out, clouds.mie_filename(r)), table, fmt="%.10e", header=header, comments="")
        print(f"  r = {r:10.4f} micron: max x = {x.max():.4g}")

    meta = {
        "refractive_index": index_info,
        "lxmie_version": importlib.metadata.version("lxmie"),
        "wavelength_range_micron": [float(lam[0]), float(lam[-1])],
        "n_lambda": args.n_lambda,
        "helios_interface_range_micron": [float(lam_lo), float(lam_hi)],
        "helios_opacity_dir": os.path.abspath(args.opacity_dir),
        "radii_micron": [float(r) for r in clouds.HELIOS_RADII],
        "columns": header.lstrip("#").split(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "chelio_git_hash": git_hash(),
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(clouds.HELIOS_RADII)} Mie files and meta.json to {args.out}")


# --- Validation (--check) ---

class _Report:
    def __init__(self):
        self.failures = []

    def check(self, ok, text):
        print(f"  [{'ok' if ok else 'FAIL'}] {text}")
        if not ok:
            self.failures.append(text)


def _is_data_row(line):
    try:
        return len([float(t) for t in line.split()]) == 7
    except ValueError:
        return False


def check_files(mie_dir, opacity_dir, report):
    print("Files")
    missing = clouds.missing_mie_files(mie_dir)
    report.check(not missing, f"all {len(clouds.HELIOS_RADII)} radius files present"
                 + (f" (missing e.g. {missing[0]})" if missing else ""))
    if missing:
        return None
    tables = {}
    bad_header = []
    for r in clouds.HELIOS_RADII:
        path = os.path.join(mie_dir, clouds.mie_filename(r))
        with open(path) as f:
            first, second = f.readline(), f.readline()
        if _is_data_row(first) or not _is_data_row(second):  # HELIOS skips exactly one line
            bad_header.append(os.path.basename(path))
        tables[r] = clouds.read_mie_file(path)
    report.check(not bad_header, "exactly one header line, then 7 numeric columns"
                 + (f" (not in {bad_header[0]}, ...)" if bad_header else ""))
    lam0 = tables[clouds.HELIOS_RADII[0]][:, 0]
    report.check(all(np.array_equal(t[:, 0], lam0) for t in tables.values()), "same wavelength grid in every file")
    report.check(bool(np.all(np.diff(lam0) > 0)), "wavelengths ascending")
    report.check(not any(np.isnan(t).any() for t in tables.values()), "no NaNs")
    if opacity_dir:
        lo, hi = clouds.helios_interface_range(opacity_dir)
        report.check(lam0[0] <= lo and lam0[-1] >= hi,
                     f"Mie grid {lam0[0]:.4g}-{lam0[-1]:.4g} micron covers HELIOS interfaces {lo:.4g}-{hi:.4g} micron")
    return tables


def check_physics(tables, report):
    print("Physics")
    rel = max(np.max(np.abs(t[:, 2] - t[:, 3] - t[:, 4]) / t[:, 2]) for t in tables.values())
    report.check(rel < 1e-6, f"C_ext = C_sca + C_abs (max rel. deviation {rel:.1e})")
    w0 = np.concatenate([t[:, 5] for t in tables.values()])
    report.check(bool(np.all((w0 >= 0) & (w0 <= 1))), f"0 <= w0 <= 1 (range {w0.min():.4g}-{w0.max():.4g})")
    g = np.concatenate([t[:, 6] for t in tables.values()])
    report.check(bool(np.all((g >= -1) & (g <= 1))), f"-1 <= g <= 1 (range {g.min():.4g}-{g.max():.4g})")
    r_big = clouds.HELIOS_RADII[-1]
    t = tables[r_big]
    q_ext = np.interp(np.log(0.5), np.log(t[:, 0]), t[:, 2]) / (np.pi * (r_big * 1e-4) ** 2)
    report.check(abs(q_ext - 2) < 0.04, f"Q_ext(r = {r_big:.0f} micron, 0.5 micron) = {q_ext:.4f} (expected 2 within 2 %)")


def check_reference(mie_dir, r_mode, sigma, report, compare):
    print(f"Size-averaged optics (r_mode = {r_mode} micron, sigma_g = {sigma})")
    lam, c_ext, c_sca, g = clouds.size_averaged_optics(mie_dir, r_mode, sigma)
    w0 = c_sca / c_ext
    rows = []
    for T, ref in REFERENCE_PLANCK_C_EXT.items():
        val = clouds.planck_mean(lam, c_ext, T)
        rows.append((f"Planck-mean C_ext, {T} K", [val], [ref]))
    for lam_ref, ref in REFERENCE_AT_LAMBDA.items():
        vals = [np.exp(np.interp(np.log(lam_ref), np.log(lam), np.log(c_ext))),
                np.interp(np.log(lam_ref), np.log(lam), w0),
                np.interp(np.log(lam_ref), np.log(lam), g)]
        rows.append((f"lambda = {lam_ref} micron: C_ext / w0 / g", vals, list(ref)))
    for label, vals, refs in rows:
        text = " / ".join(f"{v:.3g}" for v in vals)
        if compare:
            devs = [100 * (v / r - 1) for v, r in zip(vals, refs)]
            ref_text = " / ".join(f"{r:.3g}" for r in refs)
            dev_text = " / ".join(f"{d:+.1f}%" for d in devs)
            report.check(all(abs(d) <= 5 for d in devs), f"{label}: {text}  (ref. {ref_text}; {dev_text})")
        else:
            print(f"  {label}: {text}")
    return lam, c_ext, c_sca, g


def check_helios_g(mie_dir, r_mode, sigma, opacity_dir, report):
    """Runs the HELIOS Cloud class (fork in $HELIOS_PATH) and compares its size-averaged g with ours."""
    print(f"HELIOS Cloud class ({os.environ.get('HELIOS_PATH')})")
    helios_path = os.environ.get("HELIOS_PATH")
    if not helios_path:
        report.check(False, "HELIOS_PATH is set")
        return
    sys.path.insert(0, helios_path)
    try:
        from source import clouds as helios_clouds
        from source import tools as helios_tools
    finally:
        sys.path.remove(helios_path)

    def new_cloud():
        cloud = helios_clouds.Cloud()
        cloud.mie_path = [os.path.join(os.path.abspath(mie_dir), "")]
        cloud.cloud_r_mode = [float(r_mode)]
        cloud.cloud_r_std_dev = [float(sigma)]
        return cloud

    lam, c_ext, c_sca, g = clouds.size_averaged_optics(mie_dir, r_mode, sigma)

    # 1. Size-averaged values before rebinning: replace HELIOS' rebinning by the identity
    class Quant:  # minimal stand-in for HELIOS' quantity store
        opac_wave = opac_interwave = None

    real_convert = helios_tools.convert_spectrum
    helios_clouds.tls.convert_spectrum = lambda old_lambda, old_flux, *a, **kw: np.asarray(old_flux)
    try:
        cloud = new_cloud()
        cloud.calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths(0, Quant())
    finally:
        helios_clouds.tls.convert_spectrum = real_convert
    g_helios = np.asarray(cloud.g_0_one_cloud)
    c_ext_helios = np.asarray(cloud.abs_cross_one_cloud) + np.asarray(cloud.scat_cross_one_cloud)
    dev_g = np.max(np.abs(g_helios - g))
    dev_c = np.max(np.abs(c_ext_helios / c_ext - 1))
    report.check(dev_c < 1e-9, f"HELIOS size-averaged C_ext = offline (max rel. deviation {dev_c:.1e})")
    report.check(dev_g < 1e-9, f"HELIOS size-averaged g = offline g (max abs. deviation {dev_g:.1e}; "
                               f"HELIOS g range {g_helios.min():.3g}-{g_helios.max():.3g})")
    if dev_g >= 1e-9 and np.max(np.abs(g_helios)) < 1e-3:
        print("        -> HELIOS g is ~C_sca [cm^2]: the fork still has the unfixed g weighting in source/clouds.py")

    # 2. With HELIOS' own rebinning onto the opacity bins: g stays in [-1, 1] and no bin is left at zero
    if opacity_dir:
        import glob
        import h5py
        h5 = sorted(glob.glob(os.path.join(opacity_dir, "*_opac_ip*.h5")))[0]
        with h5py.File(h5, "r") as f:
            Quant.opac_wave = list(f["center wavelengths"][:])
            Quant.opac_interwave = list(f["interface wavelengths"][:])
        cloud = new_cloud()
        with contextlib.redirect_stdout(io.StringIO()):
            cloud.calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths(0, Quant())
        g_bins = np.asarray(cloud.g_0_one_cloud)
        c_bins = np.asarray(cloud.abs_cross_one_cloud) + np.asarray(cloud.scat_cross_one_cloud)
        report.check(bool(np.all(c_bins > 0)), f"rebinned onto {len(c_bins)} HELIOS bins: no bin with zero cloud opacity")
        report.check(bool(np.all((g_bins >= -1) & (g_bins <= 1))) and g_bins.max() > 0.1,
                     f"rebinned g in [-1, 1] (range {g_bins.min():.3g}-{g_bins.max():.3g})")


def plot_optics(lam, c_ext, c_sca, g, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].loglog(lam, c_ext, color="k")
    axes[0].set_ylabel(r"$\bar C_\mathrm{ext}$ [cm$^2$]")
    axes[1].semilogx(lam, c_sca / c_ext, color="C0")
    axes[1].set_ylabel(r"$\omega_0$")
    axes[1].set_ylim(0, 1.02)
    axes[2].semilogx(lam, g, color="C3")
    axes[2].set_ylabel("$g$")
    axes[2].set_ylim(min(0, g.min()) - 0.02, 1.02)
    axes[2].set_xlabel(r"$\lambda$ [$\mu$m]")
    for lam_ref, (c_ref, w_ref, g_ref) in REFERENCE_AT_LAMBDA.items():
        if "H2O_s" in title:
            axes[0].plot(lam_ref, c_ref, "o", mfc="none", color="C1")
            axes[1].plot(lam_ref, w_ref, "o", mfc="none", color="C1")
            axes[2].plot(lam_ref, g_ref, "o", mfc="none", color="C1")
    axes[0].set_title(title, fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def run_checks(args):
    report = _Report()
    print(f"Checking Mie tables in {args.out}")
    tables = check_files(args.out, args.opacity_dir, report)
    if tables is None:
        return 1
    check_physics(tables, report)
    compare = "H2O_s" in os.path.abspath(args.out) and args.r_mode == 2.0 and args.sigma == 2.0
    lam, c_ext, c_sca, g = check_reference(args.out, args.r_mode, args.sigma, report, compare)
    check_helios_g(args.out, args.r_mode, args.sigma, args.opacity_dir, report)

    name = Path(os.path.abspath(args.out)).name
    plot_path = args.plot or str(CHELIO_ROOT / "analyze" / "images" / f"cloud_optics_{name}_rmode={args.r_mode:g}_sigma={args.sigma:g}.png")
    plot_optics(lam, c_ext, c_sca, g, plot_path, f"{name}: size-averaged optics, r_mode = {args.r_mode:g} micron, sigma_g = {args.sigma:g}")

    if report.failures:
        print(f"\n{len(report.failures)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Make (or --check) HELIOS Mie tables with LX-MIE.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ri", help="refractive-index file (LX-MIE compilation format: wavelength [micron], n, k)")
    source.add_argument("--const_nk", nargs=2, type=float, metavar=("N", "K"), help="constant n and k (debugging only)")
    parser.add_argument("--out", required=True, help="output directory for the r*.dat files")
    parser.add_argument("--opacity_dir", default=default_opacity_dir(),
                        help="HELIOS opacity directory (for the wavelength range). Default: $HELIOS_PATH/input/opacity/r50_kdistr")
    parser.add_argument("--n_lambda", type=int, default=2000, help="number of wavelengths (default 2000)")
    parser.add_argument("--check", action="store_true", help="validate existing tables in --out instead of making them")
    parser.add_argument("--r_mode", type=float, default=2.0, help="--check: lognormal mode radius [micron] (default 2)")
    parser.add_argument("--sigma", type=float, default=2.0, help="--check: geometric standard deviation (default 2)")
    parser.add_argument("--plot", default=None, help="--check: figure path (default analyze/images/cloud_optics_<out>_...png)")
    args = parser.parse_args()

    if args.check:
        sys.exit(run_checks(args))
    if not (args.ri or args.const_nk):
        parser.error("give --ri <file> (or --const_nk N K for debugging), or --check")
    if not args.opacity_dir or not os.path.isdir(args.opacity_dir):
        parser.error(f"HELIOS opacity directory {args.opacity_dir!r} not found (set --opacity_dir or HELIOS_PATH)")
    try:
        make_tables(args)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
