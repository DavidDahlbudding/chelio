"""Write the HELIOS param files of the Innes et al. (2023) reproduction (grids/innes2023/innes2023.yaml).

Sun spectrum (HELIOS/star_tool/output/sun.h5, 1361.2 W/m^2 at 1 AU) at a = sqrt(1361.17 / S) AU, f = 0.25,
Earth g and radius, number of layers = n_per_dex * log10(p_surf / p_toa).
Run from the chelio root:  python3 grids/innes2023/make_param_files.py [--diffusivity 1.66]
(diffusivity factor of HELIOS' two-stream: 2 = HELIOS default, files param_S<S>_n<n>.dat; any other value gets the
suffix _d<value>, e.g. 1.66 ~ SOCRATES / Innes et al., which matches their low-S T_s within a few K)
"""
import argparse
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "test_earth_moist", "param_S1.dat")
S_SUN_1AU = 1361.17  # W/m^2: sun.h5 at 1 AU (TOA F_down of a test run; not 1364)
P_TOA_BAR = 1e-6
N_PER_DEX = 20
CASES = {1.0: range(200, 601, 50), 10.0: range(50, 101, 5)}   # p_surf [bar] -> S [W/m^2]


def set_value(text, key, value):
    pat = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*)(\S+)", re.M)
    new, n = pat.subn(lambda m: m.group(1) + str(value), text, count=1)
    if n != 1:
        raise KeyError(key)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusivity", type=float, default=2.0)
    ap.add_argument("--n_per_dex", type=int, default=N_PER_DEX, help="layers per dex (resolution tests)")
    ap.add_argument("--S", type=float, nargs="*", help="only these instellations [W/m^2]")
    args = ap.parse_args()
    suffix = "" if args.diffusivity == 2.0 else f"_d{args.diffusivity:g}"
    with open(TEMPLATE) as f:
        template = f.read()
    for p_bar, S_list in CASES.items():
        nlay = int(round(args.n_per_dex * np.log10(p_bar / P_TOA_BAR)))
        for S in S_list:
            if args.S and S not in args.S:
                continue
            t = set_value(template, "name", "innes2023")
            t = set_value(t, "manual --> orbital distance [AU]", f"{np.sqrt(S_SUN_1AU / S):.6f}")
            t = set_value(t, "number of layers", nlay)
            t = set_value(t, "TOA pressure [10^-6 bar]", f"{P_TOA_BAR * 1e6:g}")
            t = set_value(t, "BOA pressure [10^-6 bar]", f"{p_bar * 1e6:g}")
            t = set_value(t, "surface albedo", 0.12)
            t = set_value(t, "diffusivity factor", f"{args.diffusivity:g}")
            out = os.path.join(HERE, f"param_S{S}_n{nlay}{suffix}.dat")
            with open(out, "w") as f:
                f.write(t)
            print(out)


if __name__ == "__main__":
    main()
