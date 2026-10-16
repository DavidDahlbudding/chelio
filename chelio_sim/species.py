"""
Radiatively active species of a run.

`helios_inputs/all_species.dat` is the catalog of everything HELIOS has opacities for (absorbing /
scattering flags per species, CIA pairs as CIA_<A><B>). A run only loads the "active" subset of it,
since reading opacities for the on-the-fly mixing dominates the start-up time of HELIOS:

  simulation_params.active_species / --active_species
      "auto" (default) or a list of catalog species (CLI: comma-separated, e.g. "H2O,CO2,CH4,H2").

      auto, chemistry constant:           the species of constant_mixing_ratios with a non-negligible VMR
      auto, equilibrium + manual:         every catalog species made of elements with abundance > 0
      auto, equilibrium/uncoupled outgas: the species outgassed by atmodeller

CIA pairs are never listed: a pair is active if (and only if) both collision partners are.
H2O is always active (HELIOS needs it), whatever the setting.
The resulting species file is written into the run directory and handed to HELIOS.
"""
import logging
import os
import re
import sys

helios_source = os.path.join(os.environ["HELIOS_PATH"], "source")
if helios_source not in sys.path:
    sys.path.append(helios_source)
from species_database import species_lib

log = logging.getLogger(__name__)

CATALOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "helios_inputs", "all_species.dat"))

# VMRs below this are placeholders (also the floor below which create_constant_mixfile leaves species alone)
NEGLIGIBLE_VMR = 1e-29

# Always active, HELIOS fails in odd ways without H2O. Constant chemistry gets a placeholder VMR for it
# if constant_mixing_ratios has none (run_modes.resolve_run_modes).
ALWAYS_ACTIVE = ("H2O",)


def read_catalog(path=CATALOG_PATH):
    """Returns (header line, {species: catalog line}) in file order."""
    with open(path) as f:
        lines = [line.rstrip() for line in f if line.strip()]
    return lines[0], {line.split()[0]: line for line in lines[1:]}


def cia_partners(cia_species, molecules):
    """'CIA_CO2CH4' -> ('CO2', 'CH4'), with the names of the catalog `molecules` (None if a partner is not in it)."""
    by_fc_name = {species_lib[name].fc_name: name for name in molecules}
    return tuple(by_fc_name.get(fc_name) for fc_name in species_lib[cia_species].fc_name.split("&"))


def elements(species):
    """'HCN' -> {'H', 'C', 'N'}"""
    return set(re.findall(r"[A-Z][a-z]?", species_lib[species].fc_name))


def parse_active_species(value):
    """Config/CLI value -> "auto" or a list of species names."""
    if value is None or value == "auto":
        return "auto"
    if isinstance(value, str):
        value = [s.strip() for s in value.split(",") if s.strip()]
    return [str(s) for s in value]


def resolve_active_species(sim_p, outgassed_vmrs=None):
    """
    The active (non-CIA) species of a run, in catalog order.

    `outgassed_vmrs` (atmodeller species -> VMR) selects the outgassed composition for "auto";
    without it "auto" follows `chemistry`.
    """
    _, catalog = read_catalog()
    molecules = [s for s in catalog if not s.startswith("CIA_")]
    active = parse_active_species(sim_p.get("active_species"))

    if active != "auto":
        unknown = [s for s in active if s not in molecules]
        if unknown:
            raise ValueError(
                f"active_species {unknown} not in {CATALOG_PATH} (CIA pairs are added automatically). "
                f"Available: {', '.join(molecules)}"
            )
        selected = set(active)
    elif outgassed_vmrs is not None:
        selected = set(outgassed_vmrs)
    elif sim_p["chemistry"] == "constant":
        selected = {s for s, vmr in sim_p["constant_mixing_ratios"].items() if float(vmr) >= NEGLIGIBLE_VMR}
    else:
        present = {e for e, key in (("H", "a_h"), ("C", "a_c"), ("O", "a_o"), ("N", "a_n")) if float(sim_p[key]) > 0}
        selected = {s for s in molecules if elements(s) <= present}

    selected |= set(ALWAYS_ACTIVE)

    if outgassed_vmrs is not None:
        missing, source = sorted(selected - set(outgassed_vmrs)), "are not outgassed by atmodeller"
    elif sim_p["chemistry"] == "constant":
        missing, source = sorted(selected - set(sim_p["constant_mixing_ratios"])), "have no entry in constant_mixing_ratios"
    else:
        missing = []
    if missing:
        raise ValueError(f"active_species {missing} {source}.")
    return [s for s in molecules if s in selected]


def write_species_file(path, active, vmrs=None):
    """
    Writes the HELIOS species file for the active species plus the CIA pairs whose partners are both active.

    With `vmrs` (species -> constant VMR) the mixing ratios are written into the file instead of
    pointing HELIOS to the vertical mixing ratio file.
    Returns the list of species written (including CIA pairs).
    """
    header, catalog = read_catalog()
    molecules = [s for s in catalog if not s.startswith("CIA_")]
    written = []
    with open(path, "w") as f:
        f.write(header + "\n\n")
        for species, line in catalog.items():
            if species.startswith("CIA_"):
                partners = cia_partners(species, molecules)
                if not all(p in active for p in partners):
                    continue
                vmr_text = "&".join(f"{vmrs[p]:.5e}" for p in partners) if vmrs else None
            elif species in active:
                vmr_text = f"{vmrs[species]:.5e}" if vmrs else None
            else:
                continue
            if vmr_text:
                line = line.replace("file", vmr_text)
            f.write(line + "\n\n")
            written.append(species)
    return written


def setup_species_file(sim_p, run_dir, outgassed_vmrs=None, write_vmrs=False):
    """
    Resolves the active species of a run and writes <run_dir>/species.dat.
    Returns (active species without CIA pairs, path of the species file).
    """
    active = resolve_active_species(sim_p, outgassed_vmrs)
    path = os.path.join(run_dir, "species.dat")
    written = write_species_file(path, active, vmrs=outgassed_vmrs if write_vmrs else None)
    log.info(f"Active species ({sim_p.get('active_species') or 'auto'}): {', '.join(written)}")
    return active, path
