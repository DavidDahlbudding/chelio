"""
Run-mode settings shared by run_coupled.py and run_fast.py.

Three switches in `simulation_params` decide what a run does:

  chemistry                 "equilibrium": composition from GGchem, recomputed every
                                           coupling iteration from elemental abundances.
                            "constant":    fixed `constant_mixing_ratios`, only capped by
                                           saturation (no GGchem, no atmodeller).
  outgas_or_manual          (equilibrium only) where the elemental abundances come from:
                            "manual" (a_h, a_c, a_o, a_n + surface_pressure) or
                            "outgas" (atmodeller; also sets the surface pressure).
  uncoupled_outgassed_run   (outgas only) extra HELIOS run with the atmodeller gas
                            composition held fixed, i.e. without GGchem, written to
                            <out_dir>/<name>_outgassed/:
                            "none", "also" (before the coupled run) or "only" (no coupled run).

Also the cloud deck settings of run_coupled.py (CLOUD_PARAMS, resolve_clouds -> HELIOS cloud flags).
"""

import os
from pathlib import Path

from chelio_sim import clouds

CHEMISTRY_MODES = ("equilibrium", "constant")
ABUNDANCE_SOURCES = ("manual", "outgas")
UNCOUPLED_OUTGASSED_RUN_MODES = ("none", "also", "only")

# Numeric `simulation_params` that both runners accept as --<param> CLI overrides (also used by run_grid.py)
SIM_PARAMS = (
    "toa_pressure",
    "internal_temp",
    "surface_albedo",
    "melt_temp",
    "melt_frac",
    "h_ocean",
    "ctoh_ratio",
    "ntoc_ratio",
    "fO2",
    "stoc_ratio",
    "cltoc_ratio",
    "min_boa_pressure",
    "max_boa_pressure",
    "surface_pressure",
    "a_h",
    "a_c",
    "a_o",
    "a_n",
)

# Cloud deck `simulation_params` (numeric --<param> CLI overrides), plus the string --cloud_mie_path.
# run_coupled.py only (kept out of SIM_PARAMS, which run_fast.py shares: the fast RT has no clouds).
CLOUD_PARAMS = (
    "cloud_r_mode",
    "cloud_r_std_dev",
    "cloud_bottom_pressure",
    "cloud_bottom_mixing_ratio",
    "cloud_to_gas_scale_height_ratio",
)
CLOUD_PATH_PARAM = "cloud_mie_path"

# VMR given to H2O in constant chemistry if constant_mixing_ratios does not list it (too low to matter or to be
# topped up by create_constant_mixfile; HELIOS always gets H2O as an active species)
H2O_PLACEHOLDER_VMR = 1e-30

# Exit code of run_coupled.py / run_fast.py when the requested run already exists (run_grid.py reports it as skipped)
EXIT_ALREADY_DONE = 3

# Old config keys and what replaces them
RENAMED_KEYS = {
    "chemistry_mode": "`chemistry` ('ggchem' is now 'equilibrium', 'constant' is unchanged)",
    "with_ggchem": "`uncoupled_outgassed_run` (with_ggchem: False is now 'only')",
    "with_outgassed": "`uncoupled_outgassed_run` (with_outgassed: True is now 'also')",
}


def add_mode_arguments(parser):
    """Adds the CLI overrides for the run-mode switches (default: take them from the config)."""
    parser.add_argument(
        "--chemistry",
        choices=CHEMISTRY_MODES,
        default=None,
        help="'equilibrium': GGchem equilibrium chemistry; 'constant': constant mixing ratios capped by saturation. Overrides config.",
    )
    parser.add_argument(
        "--outgas_or_manual",
        choices=ABUNDANCE_SOURCES,
        default=None,
        help="Elemental abundances for equilibrium chemistry: 'manual' (a_h, a_c, a_o, a_n) or 'outgas' (atmodeller). Overrides config.",
    )
    parser.add_argument(
        "--uncoupled_outgassed_run",
        choices=UNCOUPLED_OUTGASSED_RUN_MODES,
        default=None,
        help="Extra HELIOS run with the fixed atmodeller composition (needs outgas_or_manual 'outgas'): "
             "'none', 'also' (before the coupled run) or 'only' (no coupled run). Overrides config.",
    )


    parser.add_argument(
        "--active_species",
        default=None,
        help="Species HELIOS loads opacities for: 'auto' (from the composition) or comma-separated species of "
             "helios_inputs/all_species.dat, e.g. 'H2O,CO2,CH4,H2'. CIA pairs follow automatically. Overrides config.",
    )


def add_cloud_arguments(parser):
    """CLI overrides of the cloud deck settings (run_coupled.py)."""
    parser.add_argument(
        f"--{CLOUD_PATH_PARAM}",
        default=None,
        help="Directory with the HELIOS Mie files r{:.6f}.dat (absolute or relative to the Chelio root), "
             "made with python3 -m chelio_sim.make_mie_files. Overrides config.",
    )
    for param in CLOUD_PARAMS:
        parser.add_argument(f"--{param}", type=float, default=None, help=f"Override {param} from config (cloud deck).")


def apply_cloud_overrides(sim_p, args):
    """Copies the cloud CLI overrides that were given into `sim_p` (config simulation_params)."""
    for key in (CLOUD_PATH_PARAM, *CLOUD_PARAMS):
        if getattr(args, key, None) is not None:
            sim_p[key] = getattr(args, key)


def resolve_clouds(params, p_boa=None, p_toa=None, chelio_path=None):
    """
    HELIOS cloud flags (names without the leading '-') for the cloud settings in `params` (simulation_params):
    {"number_of_cloud_decks": 0} without a cloud (cloud_bottom_mixing_ratio 0 or unset), otherwise one deck in
    HELIOS' `manual` mode (f = f_bot (p / p_bot)^(1/h - 1) above the base).

    Raises ValueError (with an explanation) if a cloud is requested that cannot work. The cloud base is only
    checked against the atmosphere if p_boa and p_toa [dyn/cm^2] are given; pass the actual BOA pressure
    (which may come from outgassing).
    """
    f_bot = float(params.get("cloud_bottom_mixing_ratio") or 0.0)
    if f_bot < 0:
        raise ValueError(f"cloud_bottom_mixing_ratio must be >= 0 (0 = no cloud), got {f_bot:g}.")
    if f_bot == 0:
        return {"number_of_cloud_decks": 0}
    if f_bot >= 1:
        raise ValueError(
            f"cloud_bottom_mixing_ratio is the number of cloud particles per gas molecule (typically ~1e-20), got {f_bot:g}."
        )

    mie_path = params.get(CLOUD_PATH_PARAM)
    if not mie_path:
        raise ValueError(
            "a cloud deck (cloud_bottom_mixing_ratio > 0) needs `cloud_mie_path`, a directory with the HELIOS Mie files "
            "(make them with: python3 -m chelio_sim.make_mie_files --ri <refractive index file> --out <dir>)."
        )
    root = chelio_path or os.environ.get("CHELIO_PATH") or Path(__file__).resolve().parent.parent
    mie_dir = os.path.abspath(os.path.join(root, os.path.expanduser(str(mie_path))))  # join keeps absolute paths
    # HELIOS concatenates directory and file name, so the directory needs the trailing "/"
    mie_dir = os.path.join(mie_dir, "")
    if not os.path.isdir(mie_dir):
        raise ValueError(f"cloud_mie_path {mie_dir} does not exist. Make the Mie files with python3 -m chelio_sim.make_mie_files.")
    missing = clouds.missing_mie_files(mie_dir)
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(clouds.HELIOS_RADII)} Mie files HELIOS needs are missing in {mie_dir} "
            f"(e.g. {missing[0]}). HELIOS reads one file per radius r = 10**np.arange(-2, 3.1, 0.1) micron, "
            "named 'r{:.6f}.dat'.format(r); make them with python3 -m chelio_sim.make_mie_files."
        )

    r_mode = float(params["cloud_r_mode"])
    sigma = float(params["cloud_r_std_dev"])
    p_bot = float(params["cloud_bottom_pressure"])
    h = float(params["cloud_to_gas_scale_height_ratio"])
    if r_mode <= 0:
        raise ValueError(f"cloud_r_mode must be > 0 micron, got {r_mode:g}.")
    if sigma <= 1:
        raise ValueError(f"cloud_r_std_dev (geometric standard deviation of the lognormal) must be > 1, got {sigma:g}.")
    if h <= 0:
        raise ValueError(f"cloud_to_gas_scale_height_ratio must be > 0, got {h:g}.")
    if p_boa is not None and p_toa is not None and not float(p_toa) < p_bot < float(p_boa):
        raise ValueError(
            f"cloud_bottom_pressure {p_bot:g} dyn/cm^2 is outside the atmosphere (TOA {float(p_toa):g}, "
            f"BOA {float(p_boa):g} dyn/cm^2). HELIOS would not find the cloud base and would fill every layer "
            "above the lowest one with cloud."
        )

    return {
        "number_of_cloud_decks": 1,
        "path_to_mie_files": mie_dir,
        "aerosol_radius_mode": r_mode,
        "aerosol_radius_geometric_std_dev": sigma,
        "cloud_mixing_ratio": "manual",
        "cloud_bottom_pressure": p_bot,
        "cloud_bottom_mixing_ratio": f_bot,
        "cloud_to_gas_scale_height_ratio": h,
    }


def parse_mixing_ratios(text):
    """Parses --constant_mixing_ratios, e.g. 'N2=0.5,CH4=0.5,CO2=0' -> {'N2': 0.5, 'CH4': 0.5, 'CO2': 0.0}."""
    mixing_ratios = {}
    for pair in text.split(","):
        try:
            key, value = pair.split("=")
            mixing_ratios[key.strip()] = float(value)
        except ValueError:
            raise ValueError(
                f"could not parse {pair!r} in --constant_mixing_ratios {text!r}. "
                "Expected comma-separated species=VMR pairs, e.g. 'N2=0.5,CH4=0.5,H2O=0.01'."
            ) from None
    return mixing_ratios


def resolve_run_modes(config, args):
    """
    Applies the CLI overrides of the run-mode switches to `config["simulation_params"]` and checks
    that the combination makes sense.

    Raises ValueError (with an explanation) if nothing sensible can be run. Settings that are merely
    ignored in the chosen mode do not stop the run; they are returned as a list of warning messages
    (to be logged once logging is set up).
    """
    for section in ("simulation_params", "coupling"):
        for old_key, new_key in RENAMED_KEYS.items():
            if old_key in (config.get(section) or {}):
                raise ValueError(
                    f"config key `{section}.{old_key}` was renamed to {new_key} in `simulation_params`. "
                    "Please update your config file."
                )

    sim_p = config["simulation_params"]
    if getattr(args, "active_species", None) is not None:
        sim_p["active_species"] = args.active_species
    defaults = {"chemistry": "equilibrium", "outgas_or_manual": "manual", "uncoupled_outgassed_run": "none"}
    for key, default in defaults.items():
        if getattr(args, key, None) is not None:
            sim_p[key] = getattr(args, key)
        # an unquoted `none`/`no` in YAML is parsed as None/False
        if sim_p.get(key) in (None, False):
            sim_p[key] = default

    for key, allowed in (
        ("chemistry", CHEMISTRY_MODES),
        ("outgas_or_manual", ABUNDANCE_SOURCES),
        ("uncoupled_outgassed_run", UNCOUPLED_OUTGASSED_RUN_MODES),
    ):
        if sim_p[key] not in allowed:
            raise ValueError(f"`{key}` must be one of {allowed}, got {sim_p[key]!r}.")

    chemistry = sim_p["chemistry"]
    outgas_or_manual = sim_p["outgas_or_manual"]
    uncoupled_outgassed_run = sim_p["uncoupled_outgassed_run"]
    relative_humidity = getattr(args, "relative_humidity", None)
    warnings = []

    # --- Uncoupled outgassed run: needs the atmodeller composition ---
    if uncoupled_outgassed_run != "none" and outgas_or_manual != "outgas":
        if uncoupled_outgassed_run == "only":
            raise ValueError(
                "uncoupled_outgassed_run 'only' runs nothing but HELIOS with the gas composition from atmodeller "
                "outgassing, which needs outgas_or_manual 'outgas'. With 'manual' there is nothing to run. "
                "Set outgas_or_manual 'outgas', or uncoupled_outgassed_run 'none' for a normal coupled run."
            )
        warnings.append(
            "uncoupled_outgassed_run 'also' is ignored: the uncoupled run uses the atmodeller outgassed composition, "
            "which needs outgas_or_manual 'outgas'. Only the coupled run is done."
        )
        sim_p["uncoupled_outgassed_run"] = uncoupled_outgassed_run = "none"

    constant_flags = [f"--{flag}" for flag in ("constant_mixing_ratios", "relative_humidity")
                      if getattr(args, flag, None) is not None]

    if uncoupled_outgassed_run == "only":
        # no coupled run, so its chemistry settings do not matter
        ignored = (["chemistry 'constant'"] if chemistry == "constant" else []) + constant_flags
        if ignored:
            warnings.append(
                f"{', '.join(ignored)} ignored: "
                "uncoupled_outgassed_run 'only' skips the coupled run and uses the atmodeller composition."
            )
        return warnings

    # --- Coupled run ---
    if chemistry == "constant":
        mixing_ratios = sim_p.get("constant_mixing_ratios")
        if not mixing_ratios:
            raise ValueError(
                "chemistry 'constant' needs `constant_mixing_ratios` in simulation_params "
                "(or --constant_mixing_ratios 'N2=0.5,CH4=0.5,...')."
            )
        vmrs = [float(v) for v in mixing_ratios.values()]
        if min(vmrs) < 0 or sum(vmrs) <= 0:
            raise ValueError(
                f"`constant_mixing_ratios` must be >= 0 with a positive sum (they are renormalised to 1), got {mixing_ratios}."
            )
        if relative_humidity is not None:
            if relative_humidity < 0:
                raise ValueError(f"--relative_humidity must be >= 0, got {relative_humidity}.")
            if not float(mixing_ratios.get("H2O", 0.0)) > 0:
                warnings.append(
                    "--relative_humidity is ignored: it only scales the H2O saturation cap, "
                    "but H2O is not in constant_mixing_ratios (or is 0)."
                )
        # H2O is always an active species (chelio_sim/species.py), so the mixfile needs a column for it
        mixing_ratios.setdefault("H2O", H2O_PLACEHOLDER_VMR)
        if outgas_or_manual == "outgas":
            used_for = ("they are only used for the uncoupled outgassed run"
                        if uncoupled_outgassed_run == "also" else "they are ignored")
            warnings.append(
                "chemistry 'constant' takes the composition from constant_mixing_ratios and the surface pressure "
                f"from surface_pressure, so the atmodeller outgassing parameters do not affect the coupled run "
                f"({used_for}). Use chemistry 'equilibrium' for a coupled outgassed atmosphere."
            )
    elif constant_flags:
        warnings.append(
            f"{', '.join(constant_flags)} ignored: only used by chemistry 'constant', "
            "but chemistry is 'equilibrium' (GGchem computes the composition, including condensation)."
        )

    return warnings


def describe_run_modes(sim_p, rt_name="HELIOS"):
    """One-line, human-readable summary of what the run will do."""
    if sim_p["uncoupled_outgassed_run"] == "only":
        return "Only the uncoupled HELIOS run with the fixed outgassed composition (no GGchem, no coupled run)."
    if sim_p["chemistry"] == "constant":
        text = f"{rt_name} coupled to constant mixing ratios (capped by saturation, no GGchem)."
    else:
        source = "atmodeller outgassing" if sim_p["outgas_or_manual"] == "outgas" else "manual elemental abundances"
        text = f"{rt_name} coupled to GGchem equilibrium chemistry ({source})."
    if sim_p["uncoupled_outgassed_run"] == "also":
        text += " Preceded by an uncoupled HELIOS run with the fixed outgassed composition."
    return text
