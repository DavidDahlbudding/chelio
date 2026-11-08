"""
HELIOS in-process ("on top of HELIOS"): runs HELIOS from Python without changing its source files.

One `HeliosSession` imports the HELIOS modules from $HELIOS_PATH once, compiles the CUDA kernels once
and then runs any number of HELIOS models one after another in the same process. On top of a plain
`python3 helios.py` run it adds:

1. Variable chemistry within one run: a VMR provider (e.g. `chem_grid.ChemGrid.provider()`) replaces
   HELIOS' FastChem-table interpolation. HELIOS calls it every 10th iteration with the current T-P
   profile, so the composition follows the temperature during the run (no coupling loop).
   Stock HELIOS resamples a FastChem table onto the opacity grid (50 K, 1/3 dex) and interpolates the
   VMRs linearly; the provider interpolates log VMRs on its own grid, so finer grids help.
2. Warm starts: start from a converged (or any) earlier solution. The radiation-only loop is skipped
   (it would push the adiabatic deep layers back towards radiative equilibrium), the fluxes are computed
   once without a temperature step (the convective adjustment needs F_up/F_down), the convective
   layers are taken over from the earlier run and their temperatures are put back on exact adiabats.
   `<name>_restart.npz` (written after every run) holds the full-precision state; HELIOS' `<name>_tp.dat`
   (conv. flags, 6 digits) and plain P-T files work as well.
3. Earlier stitching of thin radiative layers: HELIOS fills radiative gaps thinner than one pressure
   scale height between two convective zones only after 5000 iterations of the convection loop. Here
   a gap is filled as soon as one of its layers has flipped between radiative and convective
   `stitch_flip_limit` (100) times after the first `stitch_flip_start` (100) iterations (HELIOS' 5000-iteration rule stays as a fallback). Stitched layers are
   logged and written to `<name>_stitched_layers.dat`.
4. Opacity caching: the species opacity tables are read from disk once per session (keyed by file,
   mtime and size, so swapped CIA files are re-read) and kept on the GPU if they fit (`gpu_cache_fraction`
   of the free memory). Stock HELIOS re-reads them every run and re-uploads every table to the GPU
   every 10th iteration.

What is patched (only inside a process that created a HeliosSession):
  host_functions.calculate_vmr_for_all_species   -> VMR provider (stock behaviour without one)
  host_functions.mark_convective_layers          -> stitching policy (stock behaviour without one)
  read.Read.read_opac_file                       -> host cache
  computation.Compute.convection_loop and
  .calculate_total_opacity_and_scat_cross_sections_from_species -> subclass `SessionCompute`
`SessionCompute.convection_loop` is a copy of the fork's loop (HELIOS fork commit 3ae6031) with the
changes marked "chelio:"; keep it in sync if the HELIOS loop changes.

HELIOS' own "write TP profile during run" is switched off (the fork's version for non-coupled runs forces
>= 100 convection iterations and lets the radiative and convective phases overwrite each other's files);
use `write_tp_every` instead, which writes `<name>_tp_iter/{rad,conv}_<iter>.dat`.

The runs are not isolated from other HELIOS/GGchem runs on the same file system (see CLAUDE.md), but a
session never touches $HELIOS_PATH/param.dat: it passes the parameter file with -parameter_file.
"""
import contextlib
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time

import numpy as np

log = logging.getLogger(__name__)

HELIOS_PATH = os.path.abspath(os.environ["HELIOS_PATH"])

# HELIOS defaults that the session keeps unless told otherwise
HELIOS_STITCH_AFTER_ITER = 5000  # host_functions.mark_convective_layers (hardcoded there)
HELIOS_MIN_CONV_ITER = 400       # computation.convection_loop (hardcoded there)
STITCH_FLIP_LIMIT = 100          # flips of a layer in a thin radiative gap before it is stitched
STITCH_FLIP_START = 100          # convection-loop iteration from which flips are counted
COLD_START_TEMPERATURE = 500.0   # K, HELIOS' isothermal start (max(T_eff, 500 K)) and init_pt in run_coupled.py


@contextlib.contextmanager
def _in_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _argv(args):
    old = sys.argv
    sys.argv = ["helios.py", *args]
    try:
        yield
    finally:
        sys.argv = old


class _Helios:
    """The HELIOS modules (imported from $HELIOS_PATH as the package `source`)."""

    def __init__(self):
        if HELIOS_PATH not in sys.path:
            sys.path.insert(0, HELIOS_PATH)
        with _in_dir(HELIOS_PATH):
            from source import additional_heating, clouds, computation, quantities, read, realtime_plotting, write
            from source import host_functions
            from source import species_database
        self.read = read
        self.quantities = quantities
        self.hsfunc = host_functions
        self.write = write
        self.computation = computation
        self.rt_plot = realtime_plotting
        self.clouds = clouds
        self.add_heat = additional_heating
        self.species_lib = species_database.species_lib
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        self.cuda = cuda
        self.gpuarray = gpuarray


# --- patches of HELIOS module functions (installed once per process) ---

_ORIGINAL = {}
_HOST_OPACITY_CACHE = {}


def _patched_calculate_vmr_for_all_species(quant):
    provider = getattr(quant, "chelio_vmr_provider", None)
    if provider is None:
        return _ORIGINAL["calculate_vmr_for_all_species"](quant)

    quant.T_lay = quant.dev_T_lay.get()
    quant.T_int = quant.dev_T_int.get()
    quant.p_lay = quant.dev_p_lay.get()
    quant.p_int = quant.dev_p_int.get()

    names = quant.chelio_vmr_species
    vmr_lay = provider(np.asarray(quant.p_lay) * 1e-6, np.asarray(quant.T_lay[:quant.nlayer]), names)
    vmr_int = None
    if quant.iso == 0:
        vmr_int = provider(np.asarray(quant.p_int) * 1e-6, np.asarray(quant.T_int), names)

    for sp in quant.species_list:
        partners = quant.chelio_cia_partners.get(sp.name)
        if partners:
            lay = vmr_lay[partners[0]] * vmr_lay[partners[1]]
            itf = vmr_int[partners[0]] * vmr_int[partners[1]] if vmr_int is not None else None
        elif sp.name in vmr_lay:
            lay = vmr_lay[sp.name]
            itf = vmr_int[sp.name] if vmr_int is not None else None
        else:
            continue  # not provided: keeps the VMR from the species file
        sp.vmr_layer = np.array(lay, quant.fl_prec)
        if itf is not None:
            sp.vmr_interface = np.array(itf, quant.fl_prec)


def _patched_mark_convective_layers(quant, stitching):
    policy = getattr(quant, "chelio_stitch", None)
    if policy is None:
        return _ORIGINAL["mark_convective_layers"](quant, stitching)
    _ORIGINAL["mark_convective_layers"](quant, 0)
    if stitching == 1:
        policy.apply(quant)


def _cached_read_opac_file(original):
    grid_attrs = ("opac_wave", "nbin", "gauss_y", "ny", "opac_interwave", "opac_deltawave", "ktemp", "ntemp", "kpress", "npress")

    def read_opac_file(quant, name, type="premixed", read_grid_parameters=False):
        if type != "species":
            return original(quant, name, type=type, read_grid_parameters=read_grid_parameters)
        stat = os.stat(name)  # raises the IOError that HELIOS uses to try the next file name
        key = (os.path.abspath(name), stat.st_mtime_ns, stat.st_size)
        entry = _HOST_OPACITY_CACHE.get(key)
        if entry is None:
            for old_key in [k for k in _HOST_OPACITY_CACHE if k[0] == key[0]]:
                del _HOST_OPACITY_CACHE[old_key]  # file changed (e.g. swapped CIA source)
            opac = np.asarray(original(quant, name, type=type, read_grid_parameters=True))
            entry = {"opac": opac, "grid": {a: getattr(quant, a) for a in grid_attrs}}
            _HOST_OPACITY_CACHE[key] = entry
        elif read_grid_parameters:
            log.debug(f"Opacity table from cache: {name}")
        if read_grid_parameters:
            for attr, value in entry["grid"].items():
                setattr(quant, attr, value)
        quant.chelio_last_opacity_key = key
        return entry["opac"]

    return read_opac_file


def _install_patches(h):
    if _ORIGINAL:
        return
    _ORIGINAL["calculate_vmr_for_all_species"] = h.hsfunc.calculate_vmr_for_all_species
    _ORIGINAL["mark_convective_layers"] = h.hsfunc.mark_convective_layers
    h.hsfunc.calculate_vmr_for_all_species = _patched_calculate_vmr_for_all_species
    h.hsfunc.mark_convective_layers = _patched_mark_convective_layers
    h.read.Read.read_opac_file = staticmethod(_cached_read_opac_file(h.read.Read.read_opac_file))


# --- stitching of thin radiative layers ---

class StitchPolicy:
    """
    When to fill thin radiative gaps between convective zones (see the module docstring).

    flip_limit:   fill a gap (< 1 pressure scale height, HELIOS' criterion) once one of its layers flipped
                  between radiative and convective this often (None: never by flips)
    flip_start:   count flips only from this convection-loop iteration on (first tests: during the first
                  ~100 iterations layers flip all the time while the profile is still far from equilibrium)
    after_iter:   HELIOS' rule: fill all thin gaps after this many convection-loop iterations (None: never)
    surface:      also fill a flipping thin gap between the surface and the lowest convective zone. HELIOS never
                  fills that gap (a radiative layer above the surface can be real); by default it is only reported.
    """

    def __init__(self, nlayer, flip_limit=STITCH_FLIP_LIMIT, after_iter=HELIOS_STITCH_AFTER_ITER, surface=False, flip_start=STITCH_FLIP_START):
        self.flip_limit = flip_limit
        self.flip_start = flip_start
        self.after_iter = after_iter
        self.surface = surface
        self.flips = np.zeros(nlayer + 1, int)
        self.previous = None
        self.latched = set()       # layers (HELIOS indices, nlayer = surface) that reached flip_limit
        self.filled_now = set()    # layers filled by the latest marking
        self.events = []           # (iteration, layer, reason)
        self.surface_warned = False
        self.helios_rule_active = False

    def count_flips(self, quant):
        """Once per convection-loop iteration, after the loop's own marking."""
        current = np.array(quant.conv_layer, int)
        if self.previous is not None and quant.iter_value >= self.flip_start:
            self.flips += current != self.previous
        self.previous = current
        if self.flip_limit is not None:
            for layer in np.flatnonzero(self.flips >= self.flip_limit):
                if layer not in self.latched:
                    self.latched.add(int(layer))
                    log.debug(f"Layer {layer} flipped {self.flips[layer]} times (iteration {quant.iter_value})")

    def _column(self, quant):
        """Bottom-to-top column: index 0 = surface ghost layer, k = HELIOS layer k-1."""
        n = quant.nlayer
        conv = np.concatenate(([quant.conv_layer[n]], quant.conv_layer[:n]))
        press = np.concatenate(([quant.p_int[0]], quant.p_lay))
        helios_index = np.concatenate(([n], np.arange(n)))
        return conv, press, helios_index

    def _gaps(self, quant):
        """Radiative gaps as (column indices, bounded below by a convective layer?, thin?)."""
        conv, press, _ = self._column(quant)
        gaps, k = [], 0
        while k < len(conv):
            if conv[k] == 0:
                start = k
                while k < len(conv) and conv[k] == 0:
                    k += 1
                if k == len(conv):
                    break  # radiative up to the top: not a gap
                p_top = press[k]  # lowest layer of the convective zone above
                p_bot = press[start - 1] if start > 0 else press[0]
                gaps.append((list(range(start, k)), start > 0, p_top / p_bot > 1 / np.e))
            else:
                k += 1
        return gaps

    def apply(self, quant):
        self.filled_now = set()
        before = np.array(quant.conv_layer, int)
        if self.after_iter is not None and quant.iter_value > self.after_iter:
            if not self.helios_rule_active:
                log.info(f"Iteration {quant.iter_value} > {self.after_iter}: HELIOS fills all thin radiative gaps from now on")
                self.helios_rule_active = True
            _ORIGINAL["stitching"](quant)
            self._record(quant, before, f"iter>{self.after_iter}")
            before = np.array(quant.conv_layer, int)

        if not self.latched:
            return
        _, _, helios_index = self._column(quant)
        for column_idx, bounded_below, thin in self._gaps(quant):
            layers = {int(helios_index[c]) for c in column_idx}
            if not thin or not (layers & self.latched):
                continue
            if not bounded_below and not self.surface:
                if not self.surface_warned:
                    log.warning(f"Thin radiative gap above the surface keeps flipping (layers {sorted(layers)}); "
                                "not stitched (StitchPolicy.surface = False)")
                    self.surface_warned = True
                continue
            for layer in layers:
                quant.conv_layer[layer] = 1
        self._record(quant, before, f"flips>={self.flip_limit}")

    def _record(self, quant, before, reason):
        filled = np.flatnonzero((np.array(quant.conv_layer, int) == 1) & (before == 0))
        for layer in filled:
            layer = int(layer)
            if layer not in {e[1] for e in self.events if e[2] == reason}:
                self.events.append((int(quant.iter_value), layer, reason))
                log.info(f"Stitched layer {layer} at iteration {quant.iter_value} ({reason})")
        self.filled_now |= set(int(layer) for layer in filled)


# --- diagnostics ---

def radiative_imbalance(quant):
    """
    (global, max over radiative layers) relative flux imbalance, as HELIOS measures it:
    |F_int + F_heat + F_smooth - F_net| / (F_down(BOA) + F_int); global at the TOA interface.
    """
    n = quant.nlayer
    norm = quant.F_down_tot[n] + quant.F_intern
    glob = abs(quant.F_intern + quant.F_add_heat_sum[n - 1] - quant.F_net[n]) / norm
    worst = 0.0
    for i in range(n + 1):
        if quant.conv_layer[i] == 0:
            if i < n:
                d = abs(quant.F_intern + quant.F_add_heat_sum[i] + quant.F_smooth_sum[i] - quant.F_net[i + 1])
            else:
                d = abs(quant.F_intern - quant.F_net[0])
            worst = max(worst, d / norm)
    return float(glob), float(worst)


# --- compute subclass ---

def _make_compute_class(h):
    comp = h.computation
    hsfunc = h.hsfunc
    gpuarray = h.gpuarray
    cuda = h.cuda

    class SessionCompute(comp.Compute):

        def __init__(self, session):
            super().__init__()
            self.session = session

        # chelio: device-cached species opacities (stock: re-uploaded every call, i.e. every 10th iteration)
        def calculate_total_opacity_and_scat_cross_sections_from_species(self, quant):
            for s in range(len(quant.species_list)):
                sp = quant.species_list[s]
                quant.dev_vmr_spec_lay = gpuarray.to_gpu(sp.vmr_layer)
                if quant.iso == 0:
                    quant.dev_vmr_spec_int = gpuarray.to_gpu(sp.vmr_interface)

                if sp.absorbing == "yes":
                    quant.dev_opacity_spec_pretab = self.session._device_opacity(sp)
                    self.interpolate_species_opac(quant)
                    self.add_to_mixed_opacity(quant, sp.weight, s)

                if sp.scattering == "yes":
                    if sp.name == "H2O":
                        quant.dev_scat_cross_spec_lay = gpuarray.to_gpu(np.zeros(quant.nbin * quant.nlayer, quant.fl_prec))
                        if quant.iso == 0:
                            quant.dev_scat_cross_spec_int = gpuarray.to_gpu(np.zeros(quant.nbin * quant.ninterface, quant.fl_prec))
                        self.calculate_H2O_Rayleigh_scattering(quant, s)
                    else:
                        quant.dev_scat_cross_spec_lay = gpuarray.to_gpu(sp.scat_cross_sect_layer)
                        if quant.iso == 0:
                            quant.dev_scat_cross_spec_int = gpuarray.to_gpu(sp.scat_cross_sect_interface)
                    self.add_to_mixed_scat_cross_sect(quant)

        def update_atmosphere(self, quant):
            """The every-10th-iteration block of the loops: composition, opacities, transmission, heights, beam."""
            if quant.opacity_mixing == "premixed":
                self.interpolate_opacities_and_scattering_cross_sections(quant)
                self.interpolate_meanmolmass(quant)
            elif quant.opacity_mixing == "on-the-fly":
                hsfunc.calculate_vmr_for_all_species(quant)
                hsfunc.calculate_meanmolecularmass(quant)
                hsfunc.nullify_opac_scat_arrays(quant)
                self.calculate_total_opacity_and_scat_cross_sections_from_species(quant)
            if quant.clouds == 1:
                self.calc_total_g_0_of_gas_and_clouds(quant)
            self.calculate_transmission(quant)
            self.calculate_delta_z(quant)
            quant.delta_z_lay = quant.dev_delta_z_lay.get()
            hsfunc.calculate_height_z(quant)
            quant.dev_z_lay = gpuarray.to_gpu(quant.z_lay)
            self.calculate_direct_beamflux(quant)

        def compute_fluxes(self, quant):
            if quant.flux_calc_method == "iteration":
                self.populate_spectral_flux_iteratively(quant)
            elif quant.flux_calc_method == "matrix":
                self.solve_for_spectral_fluxes_via_matrix(quant)
            else:
                raise RuntimeError(f"Flux calculation method {quant.flux_calc_method!r} unclear")
            self.integrate_flux(quant)

        def prime_fluxes(self, quant):
            """chelio: one flux evaluation without temperature step (warm start instead of the radiation loop)."""
            quant.iter_value = np.int32(0)
            quant.p_lay = quant.dev_p_lay.get()
            quant.p_int = quant.dev_p_int.get()
            self.interpolate_temperatures(quant)
            self.interpolate_planck(quant)
            self.update_atmosphere(quant)
            self.compute_fluxes(quant)

        def convection_loop(self, quant, write, read, rt_plot):
            """Copy of the fork's convection loop; chelio changes are marked."""
            opts = self.session._run_opts
            policy = getattr(quant, "chelio_stitch", None)

            if quant.singlewalk == 0 and quant.convection == 1:

                self.interpolate_kappa_and_cp(quant)
                quant.T_lay = quant.dev_T_lay.get()
                quant.p_lay = quant.dev_p_lay.get()
                quant.p_int = quant.dev_p_int.get()
                quant.kappa_lay = quant.dev_kappa_lay.get()
                if quant.iso == 0:
                    quant.kappa_int = quant.dev_kappa_int.get()
                    hsfunc.conv_check(quant)
                    hsfunc.mark_convective_layers(quant, stitching=0)

                # chelio: warm start -> convective layers of the earlier run, temperatures back on exact adiabats
                warm_conv = opts.get("warm_conv_layer")
                if warm_conv is not None:
                    quant.conv_layer = np.array(warm_conv, np.int32)
                    quant.conv_unstable = np.zeros(quant.nlayer + 1, np.int32)
                    if quant.opacity_mixing == "on-the-fly":
                        hsfunc.calculate_meanmolecularmass(quant)
                    else:
                        self.interpolate_meanmolmass(quant)
                    quant.c_p_lay = quant.dev_c_p_lay.get()
                    quant.meanmolmass_lay = quant.dev_meanmolmass_lay.get()
                    quant.F_net = quant.dev_F_net.get()
                    quant.F_up_tot = quant.dev_F_up_tot.get()
                    quant.F_down_tot = quant.dev_F_down_tot.get()
                    quant.F_smooth_sum = quant.dev_F_smooth_sum.get()
                    diag = quant.chelio_diag
                    diag["imbalance_after_prime"] = radiative_imbalance(quant)
                    T_before = np.array(quant.T_lay, float)
                    hsfunc.conv_correct(quant, fudging=0)
                    dT = np.array(quant.T_lay, float) - T_before
                    diag["warm_adiabat_max_abs_dT"] = float(np.max(np.abs(dT)))
                    diag["warm_adiabat_dT_surface"] = float(dT[quant.nlayer])
                    quant.dev_T_lay = gpuarray.to_gpu(quant.T_lay)
                    self.interpolate_temperatures(quant)
                    self.interpolate_kappa_and_cp(quant)
                    quant.kappa_lay = quant.dev_kappa_lay.get()
                    if quant.iso == 0:
                        quant.kappa_int = quant.dev_kappa_int.get()
                    # fluxes of the corrected profile (diagnostics; the loop recomputes them anyway)
                    self.interpolate_planck(quant)
                    self.update_atmosphere(quant)
                    self.compute_fluxes(quant)
                    quant.F_net = quant.dev_F_net.get()
                    quant.F_down_tot = quant.dev_F_down_tot.get()
                    diag["imbalance_after_adiabat"] = radiative_imbalance(quant)
                    log.info(f"Warm start: imbalance (global, max radiative layer) after the flux pass "
                             f"{diag['imbalance_after_prime'][0]:.2e}, {diag['imbalance_after_prime'][1]:.2e}; after putting "
                             f"the convective zones on adiabats (max |dT| {diag['warm_adiabat_max_abs_dT']:.3g} K) "
                             f"{diag['imbalance_after_adiabat'][0]:.2e}, {diag['imbalance_after_adiabat'][1]:.2e}")

                # chelio: a warm start enters the loop with an exactly adiabatic (= stable) profile
                condition = sum(quant.conv_unstable) > 0 or (warm_conv is not None and sum(quant.conv_layer) > 0)

                start_total = cuda.Event()
                end_total = cuda.Event()
                start_total.record()

                quant.rad_convergence_limit *= 0.1**np.sum(quant.iter_value >= np.array(quant.crit_relaxation_numbers))

                quant.iter_value = np.int32(0)
                time_loop = 0.0
                start_loop = cuda.Event()
                end_loop = cuda.Event()

                if condition:
                    print("\nConvectively unstable layers found. Starting convective adjustment")
                else:
                    print("\nAll layers convectively stable. No convective adjustment necessary.\n")

                quant.F_net = quant.dev_F_net.get()
                quant.F_up_tot = quant.dev_F_up_tot.get()
                quant.F_down_tot = quant.dev_F_down_tot.get()

                min_iter = opts.get("min_conv_iter", HELIOS_MIN_CONV_ITER)
                # chelio: no forced >= 100 iterations for coupling iterations > 0 (fork), see min_conv_iter
                while condition:

                    if quant.iter_value % 100 == 0:
                        start_loop.record()
                        print("\nWe are running \"" + quant.name + "\" at iteration step nr. : " + str(quant.iter_value))
                        if quant.iter_value > 99:
                            print("Time for the last 100 steps [s]: {:.2f}".format(time_loop * 1e-3))

                    self.interpolate_temperatures(quant)

                    if quant.iter_value % 10 == 0:
                        if quant.opacity_mixing == "premixed":
                            self.interpolate_meanmolmass(quant)
                        elif quant.opacity_mixing == "on-the-fly":
                            hsfunc.calculate_vmr_for_all_species(quant)
                            hsfunc.calculate_meanmolecularmass(quant)

                    self.interpolate_kappa_and_cp(quant)
                    quant.kappa_lay = quant.dev_kappa_lay.get()
                    if quant.iso == 0:
                        quant.kappa_int = quant.dev_kappa_int.get()
                    quant.c_p_lay = quant.dev_c_p_lay.get()
                    quant.meanmolmass_lay = quant.dev_meanmolmass_lay.get()
                    quant.T_lay = quant.dev_T_lay.get()
                    quant.F_smooth_sum = quant.dev_F_smooth_sum.get()
                    hsfunc.convective_adjustment(quant)
                    quant.dev_T_lay = gpuarray.to_gpu(quant.T_lay)

                    self.interpolate_temperatures(quant)
                    self.interpolate_planck(quant)

                    if quant.iter_value % 10 == 0:
                        self.update_atmosphere(quant)
                    self.compute_fluxes(quant)

                    quant.F_net = quant.dev_F_net.get()
                    quant.F_down_tot = quant.dev_F_down_tot.get()
                    quant.F_up_tot = quant.dev_F_up_tot.get()
                    quant.F_net_diff = quant.dev_F_net_diff.get()

                    self.interpolate_kappa_and_cp(quant)
                    quant.kappa_lay = quant.dev_kappa_lay.get()
                    if quant.iso == 0:
                        quant.kappa_int = quant.dev_kappa_int.get()
                    quant.T_lay = quant.dev_T_lay.get()

                    hsfunc.mark_convective_layers(quant, stitching=1)
                    if policy is not None:  # chelio
                        policy.count_flips(quant)

                    if quant.physical_tstep != 0:
                        break

                    quant.F_smooth_sum = quant.dev_F_smooth_sum.get()
                    condition = not (hsfunc.check_for_radiative_eq(quant)) or (quant.iter_value < min_iter) or (sum(quant.conv_layer) == 0)

                    if quant.iter_value % 100 == 1:
                        hsfunc.give_feedback_on_convergence(quant)
                    if quant.iter_value % 100 == 0:  # chelio: convergence history for <name>_session.json
                        g, w = radiative_imbalance(quant)
                        quant.chelio_diag["history"].append([int(quant.iter_value), g, w, int(sum(quant.conv_layer[:quant.nlayer])),
                                                             float(quant.T_lay[quant.nlayer])])

                    if condition:
                        if quant.iter_value % quant.n_plot == 0 and quant.realtime_plot == 1:
                            rt_plot.plot_tp_and_flux(quant)

                        if quant.add_heating == 1:
                            if quant.iter_value % 10 == 0:
                                hsfunc.calc_add_heating_flux(quant)
                                quant.dev_F_add_heat_lay = gpuarray.to_gpu(quant.F_add_heat_lay)
                                quant.dev_F_add_heat_sum = gpuarray.to_gpu(quant.F_add_heat_sum)

                        quant.dev_conv_layer = gpuarray.to_gpu(quant.conv_layer)
                        quant.dev_marked_red = gpuarray.to_gpu(quant.marked_red)
                        self.conv_temp_iteration(quant)

                        quant.T_lay = quant.dev_T_lay.get()

                        if quant.iter_value % 100 == 99:
                            end_loop.record()
                            end_loop.synchronize()
                            time_loop = start_loop.time_till(end_loop)

                        quant.iter_value += 1
                        quant.iter_value = np.int32(quant.iter_value)

                    # chelio: own intermediate T-P output instead of HELIOS' "write TP profile during run"
                    every = opts.get("write_tp_every") or 0
                    if every and quant.iter_value % every == 0:
                        self.session._write_intermediate_tp(quant, read, "conv")

                    if quant.iter_value in quant.crit_relaxation_numbers:
                        hsfunc.relax_radiative_convergence_criterion(quant)

                    # chelio: stop instead of SystemExit, the session still writes the output
                    if quant.iter_value > quant.max_nr_iterations:
                        write.write_abort_file(quant, read)
                        print("\nRun exceeds allowed maximum allowed number of iteration steps. Aborting...")
                        quant.chelio_aborted = True
                        break

                end_total.record()
                end_total.synchronize()
                time_total = start_total.time_till(end_total)

                print("\nTime for rad.-conv. iteration [s]: {:.2f}".format(time_total * 1e-3))
                print("Total number of iterative steps: " + str(quant.iter_value))
                quant.chelio_conv_iter = int(quant.iter_value)

    return SessionCompute


# --- initial states ---

def load_initial_state(path):
    """
    Earlier solution to start from. Returns dict(p_bar, T, conv_layer or None, source):
      p_bar, T: BOA interface first, then the layers bottom to top (like the HELIOS T-P file for coupling)
      conv_layer: HELIOS conv_layer flags (layers 0..n-1, surface at n), or None
    Accepts <name>_restart.npz (exact), HELIOS <name>_tp.dat (with conv. flags) or any file with P [bar], T [K]
    columns and one header line (e.g. <name>_tp_coupling_<i>.dat).
    """
    path = str(path)
    if path.endswith(".npz"):
        with np.load(path) as data:
            p = np.concatenate(([data["p_int"][0]], data["p_lay"])) * 1e-6
            T = np.concatenate(([data["T_lay"][-1]], data["T_lay"][:-1]))
            inputs = json.loads(str(data["inputs"])) if "inputs" in data.files else None
            return {"p_bar": p, "T": T, "conv_layer": np.array(data["conv_layer"]), "source": path, "inputs": inputs}

    with open(path) as f:
        first = f.readline()
    if first.startswith("This file contains the corresponding layer temperatures"):
        rows = []
        with open(path) as f:
            lines = f.read().splitlines()[2:]
        for line in lines:
            col = line.split()
            if len(col) < 3:
                continue
            flag = col[6] if len(col) > 6 and col[6] not in ("not_calculated",) else None
            rows.append((float(col[2]) * 1e-6, float(col[1]), None if flag is None else int(float(flag))))
        p = np.array([r[0] for r in rows])
        T = np.array([r[1] for r in rows])
        flags = [r[2] for r in rows]
        conv = None
        if all(f is not None for f in flags):
            conv = np.array(flags[1:] + flags[:1], np.int32)  # surface (BOA row) last, as in HELIOS
        return {"p_bar": p, "T": T, "conv_layer": conv, "source": path}

    data = np.loadtxt(path, skiprows=1)
    return {"p_bar": data[:, 0], "T": data[:, 1], "conv_layer": None, "source": path}


def _same_grid(p_a, p_b, rtol=1e-4):
    return len(p_a) == len(p_b) and np.allclose(p_a, p_b, rtol=rtol)


# --- the session ---

class HeliosSession:
    """
    Runs HELIOS models in this process (see the module docstring). Typical use:

        session = HeliosSession()
        result = session.run(params, vmr_provider=grid.provider(), init="output/x/x_restart.npz")

    gpu_cache_fraction: keep species opacity tables on the GPU while their total size stays below this
                        fraction of the free GPU memory at session start (0 = stock behaviour).
    """

    def __init__(self, gpu_cache_fraction=0.5):
        self.h = _Helios()
        _install_patches(self.h)
        _ORIGINAL.setdefault("stitching", self.h.hsfunc.stitching_convective_zone_holes)
        with _in_dir(HELIOS_PATH):
            self.compute = _make_compute_class(self.h)(self)
        free, total = self.h.cuda.mem_get_info()
        self.gpu_cache_budget = gpu_cache_fraction * free
        self._device_cache = {}
        self._device_cache_bytes = 0
        self._run_opts = {}
        self._current = None
        log.info(f"HELIOS session ready (GPU: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free, "
                 f"opacity cache budget {self.gpu_cache_budget / 2**30:.1f} GiB)")

    # -- opacity cache on the GPU --
    def _device_opacity(self, sp):
        key = getattr(sp, "chelio_opacity_key", None)
        if key is None:
            return self.h.gpuarray.to_gpu(sp.opacity_pretab)
        dev = self._device_cache.get(key)
        if dev is not None:
            return dev
        for old in [k for k in self._device_cache if k[0] == key[0]]:
            self._device_cache_bytes -= self._device_cache.pop(old).nbytes
        dev = self.h.gpuarray.to_gpu(sp.opacity_pretab)
        if self._device_cache_bytes + dev.nbytes <= self.gpu_cache_budget:
            self._device_cache[key] = dev
            self._device_cache_bytes += dev.nbytes
        return dev

    def _write_intermediate_tp(self, quant, read, phase):
        T = quant.dev_T_lay.get()
        directory = os.path.join(read.output_path, quant.name, f"{quant.name}_tp_iter")
        os.makedirs(directory, exist_ok=True)
        p = np.concatenate(([quant.p_int[0]], quant.p_lay)) * 1e-6
        T = np.concatenate(([T[quant.nlayer]], T[:quant.nlayer]))
        np.savetxt(os.path.join(directory, f"{phase}_{int(quant.iter_value)}.dat"), np.column_stack([p, T]),
                   fmt="%.12e", header="press.[bar] temp.[K]", comments="")

    @staticmethod
    def helios_args(params):
        """HELIOS command-line options from a parameter dict (keys = HELIOS option names without '-')."""
        args = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "yes" if value else "no"
            args += [f"-{key}", str(value)]
        return args

    def run(self, params, vmr_provider=None, vmr_species=None, init=None, cold_start_temperature=COLD_START_TEMPERATURE,
            min_conv_iter=HELIOS_MIN_CONV_ITER, rad_max_iter=None, write_tp_every=0,
            stitch_flip_limit=STITCH_FLIP_LIMIT, stitch_after_iter=HELIOS_STITCH_AFTER_ITER, stitch_surface=False,
            stitch_flip_start=STITCH_FLIP_START):
        """
        One HELIOS model. `params`: HELIOS command-line options (e.g. name, output_directory, parameter_file,
        boa_pressure, internal_temperature, path_to_species_file, kappa_file_path, maximum_number_of_iterations).
        The session sets opacity_mixing on-the-fly, coupling_mode no and write_tp_profile_during_run 0.

        vmr_provider(p_bar, T, species) -> {species: VMR array}: composition as function of the T-P profile;
            vmr_species: species to ask for (default: all non-CIA species of the species file)
        init: earlier solution (see load_initial_state) for a warm start; None = isothermal cold start
        rad_max_iter: cold start only, cap of the radiation-only loop; then continue with the convection loop
            (the fork's started_convection behaviour). Default: maximum_number_of_iterations.
        stitch_flip_limit, stitch_after_iter, stitch_surface, stitch_flip_start: StitchPolicy (stitch_flip_limit None and
            stitch_after_iter 5000 = HELIOS)
        Returns a dict with convergence, iteration counts, timings, the final profile and stitching info,
        also written to <output_directory>/<name>/<name>_session.json.
        """
        h = self.h
        t0 = time.time()
        params = dict(params)
        params.update({"opacity_mixing": "on-the-fly", "coupling_mode": "no", "write_tp_profile_during_run": 0})
        params.setdefault("start_from_provided_tp_profile", "no")
        init_state = load_initial_state(init) if init else None
        kappa_file = params.get("kappa_file_path")
        inputs = {"kappa_file_sha1": hashlib.sha1(open(kappa_file, "rb").read()).hexdigest()
                  if kappa_file and os.path.exists(kappa_file) else None,
                  "vmr_provider": repr(vmr_provider) if vmr_provider is not None else None}

        reader = h.read.Read()
        keeper = h.quantities.Store()
        writer = h.write.Write()
        plotter = h.rt_plot.Plot()
        fogger = h.clouds.Cloud()
        result = {"name": params.get("name"), "warm_start": init_state["source"] if init_state else None}
        if init_state and init_state.get("inputs"):
            differ = [k for k, v in inputs.items() if init_state["inputs"].get(k) != v]
            result["warm_start_inputs_differ"] = differ
            if differ:
                log.warning(f"Warm start: {differ} differ from the earlier run, so it is not an equilibrium of this model")

        try:
            with _in_dir(HELIOS_PATH), _argv(self.helios_args(params)):
                reader.read_param_file_and_command_line(keeper, fogger)

                if keeper.opacity_mixing == "premixed":
                    reader.load_premixed_opacity_table(keeper)
                else:
                    reader.read_species_file(keeper)
                    reader.read_species_opacities(keeper)
                    self._tag_opacity_keys(keeper, reader)
                    reader.read_species_scat_cross_sections(keeper)
                    reader.read_species_mixing_ratios(keeper)
                self._setup_vmr_provider(keeper, vmr_provider, vmr_species)

                reader.read_kappa_table_or_use_constant_kappa(keeper)
                reader.read_or_fill_surf_albedo_array(keeper)
                keeper.dimensions()
                reader.read_star(keeper)
                h.hsfunc.planet_param(keeper, reader)
                h.hsfunc.set_up_numerical_parameters(keeper)
                h.hsfunc.construct_grid(keeper)
                h.hsfunc.initial_temp(keeper, reader)
                warm_conv = self._set_initial_temperature(keeper, init_state, cold_start_temperature)
                result["warm_start_exact_grid"] = warm_conv is not None

                if keeper.approx_f == 1 and keeper.planet_type == "rocky":
                    h.hsfunc.approx_f_from_formula(keeper, reader)
                h.hsfunc.calc_F_intern(keeper)
                h.add_heat.load_heating_terms_or_not(keeper)
                fogger.cloud_pre_processing(keeper)

                keeper.create_zero_arrays()
                keeper.convert_input_list_to_array()
                keeper.copy_host_to_device()
                keeper.allocate_on_device()

                keeper.chelio_stitch = StitchPolicy(keeper.nlayer, stitch_flip_limit, stitch_after_iter, stitch_surface, stitch_flip_start)
                keeper.chelio_aborted = False
                keeper.chelio_conv_iter = 0
                keeper.chelio_diag = {"history": []}
                keeper.chelio_inputs = inputs
                self._run_opts = {"min_conv_iter": min_conv_iter, "write_tp_every": write_tp_every,
                                  "warm_conv_layer": warm_conv if init_state else None}
                t_setup = time.time()

                self.compute.construct_planck_table(keeper)
                self.compute.correct_incident_energy(keeper)

                rad_iter = 0
                if init_state is None:
                    # cold start: HELIOS' radiation loop; at its iteration cap continue with convection (fork behaviour)
                    reader.started_convection = 1
                    saved_max = keeper.max_nr_iterations
                    if rad_max_iter is not None:
                        keeper.max_nr_iterations = np.int32(rad_max_iter)
                    self.compute.radiation_loop(keeper, writer, reader, plotter)
                    rad_iter = int(keeper.iter_value)
                    keeper.max_nr_iterations = saved_max
                    if write_tp_every:
                        self._write_intermediate_tp(keeper, reader, "rad")
                else:
                    self.compute.prime_fluxes(keeper)
                    if warm_conv is None or keeper.convection == 0:
                        # no flags from the earlier run: HELIOS marks the layers itself at the loop start
                        self._run_opts["warm_conv_layer"] = None
                result["rad_iter"] = rad_iter

                self.compute.convection_loop(keeper, writer, reader, plotter)
                t_loops = time.time()

                self.compute.integrate_optdepth_transmission(keeper)
                self.compute.calculate_contribution_function(keeper)
                if keeper.convection == 1:
                    self.compute.interpolate_entropy(keeper)
                    self.compute.interpolate_phase_state(keeper)
                self.compute.calculate_mean_opacities(keeper)
                self.compute.integrate_beamflux(keeper)

                keeper.copy_device_to_host()
                h.hsfunc.calculate_conv_flux(keeper)
                h.hsfunc.calc_F_ratio(keeper)
                self._write_helios_output(keeper, reader, writer)
                if keeper.approx_f == 1:
                    h.hsfunc.calc_tau_lw_sw(keeper, reader)
                h.hsfunc.success_message(keeper)

                result.update(self._finish(keeper, reader, t0, t_setup, t_loops))
                write_session_info(self._last_run_dir, keeper.name, result)
        except SystemExit as e:
            raise RuntimeError(f"HELIOS stopped (SystemExit {e.code}) in run {params.get('name')}") from None
        finally:
            del keeper
            gc.collect()
        return result

    # -- helpers of run() --
    def _tag_opacity_keys(self, keeper, reader):
        for sp in keeper.species_list:
            sp.chelio_opacity_key = None
            if sp.absorbing != "yes":
                continue
            for suffix in ("_opac_ip_kdistr.h5", "_opac_ip.h5", "_opac_ip_sampling.h5"):
                path = os.path.abspath(reader.opacity_path + sp.name + suffix)
                if os.path.exists(path):
                    stat = os.stat(path)
                    sp.chelio_opacity_key = (path, stat.st_mtime_ns, stat.st_size, str(keeper.fl_prec))
                    break

    def _setup_vmr_provider(self, keeper, provider, species):
        keeper.chelio_vmr_provider = provider
        if provider is None:
            return
        lib = self.h.species_lib
        molecules = [sp.name for sp in keeper.species_list if not sp.name.startswith("CIA")]
        keeper.chelio_vmr_species = list(species) if species else molecules
        by_fc = {lib[name].fc_name: name for name in molecules if name in lib}
        keeper.chelio_cia_partners = {}
        for sp in keeper.species_list:
            if sp.name.startswith("CIA"):
                partners = tuple(by_fc.get(fc) for fc in lib[sp.name].fc_name.split("&"))
                if None in partners:
                    raise ValueError(f"{sp.name}: collision partner not among the species {molecules}")
                keeper.chelio_cia_partners[sp.name] = partners

    def _set_initial_temperature(self, keeper, init_state, cold_T):
        """Sets keeper.T_lay; returns the earlier run's conv_layer if the grids match (exact warm start)."""
        p_own = np.concatenate(([keeper.p_int[0]], keeper.p_lay)) * 1e-6
        if init_state is None:
            keeper.T_lay = np.full(keeper.nlayer + 1, float(cold_T))
            print(f"\nchelio: isothermal start at {cold_T:g} K.")
            return None
        if _same_grid(init_state["p_bar"], p_own):
            T = np.array(init_state["T"], float)
            conv = init_state["conv_layer"]
        else:
            logp = np.log10(init_state["p_bar"])
            order = np.argsort(logp)
            T = np.interp(np.log10(p_own), logp[order], np.asarray(init_state["T"])[order])
            conv = None
            log.info("Warm start: other pressure grid, T interpolated, convective layers are marked anew")
        keeper.T_lay = np.append(T[1:], T[0])
        print(f"\nchelio: warm start from {init_state['source']}.")
        return conv

    def _write_helios_output(self, keeper, reader, writer):
        # HELIOS' create_output_dir_and_copy_param_file needs a param file path relative to $HELIOS_PATH
        run_dir = os.path.join(reader.output_path, keeper.name)
        os.makedirs(run_dir, exist_ok=True)
        shutil.copyfile(os.path.join(HELIOS_PATH, reader.param_file), os.path.join(run_dir, f"{keeper.name}_param.dat"))
        for method in ("write_colmass_mu_cp_entropy", "write_integrated_flux", "write_downward_spectral_flux",
                       "write_upward_spectral_flux", "write_TOA_flux_eclipse_depth", "write_direct_spectral_beam_flux",
                       "write_planck_interface", "write_planck_center", "write_tp", "write_tp_cut", "write_opacities",
                       "write_cloud_mixing_ratio", "write_cloud_opacities", "write_Rayleigh_cross_sections",
                       "write_cloud_scat_cross_sections", "write_g_0", "write_transmission", "write_opt_depth",
                       "write_cloud_opt_depth", "write_trans_weight_function", "write_contribution_function",
                       "write_mean_extinction", "write_flux_ratio_only", "write_phase_state", "write_surface_albedo",
                       "write_criterion_warning_file"):
            getattr(writer, method)(keeper, reader)

    def _finish(self, keeper, reader, t0, t_setup, t_loops):
        run_dir = os.path.join(reader.output_path, keeper.name)
        n = keeper.nlayer
        T_lay = np.array(keeper.T_lay, float)
        conv_layer = np.array(keeper.conv_layer, np.int32) if keeper.conv_layer is not None else np.zeros(n + 1, np.int32)
        p_lay, p_int = np.array(keeper.p_lay, float), np.array(keeper.p_int, float)
        np.savez(os.path.join(run_dir, f"{keeper.name}_restart.npz"), T_lay=T_lay, p_lay=p_lay, p_int=p_int,
                 conv_layer=conv_layer, internal_temperature=float(keeper.T_intern), g=float(keeper.g),
                 inputs=json.dumps(keeper.chelio_inputs))

        # T-P file in the format of the fork's coupling output (bar, K), full precision
        p = np.concatenate(([p_int[0]], p_lay)) * 1e-6
        T = np.concatenate(([T_lay[n]], T_lay[:n]))
        np.savetxt(os.path.join(run_dir, f"{keeper.name}_tp_coupling_0.dat"), np.column_stack([p, T]),
                   fmt="%-24.12e%.12e", header=f"{'press.[bar]':<24}temp.[K]", comments="")

        policy = keeper.chelio_stitch
        stitched_final = sorted(policy.filled_now)
        if policy.events:
            with open(os.path.join(run_dir, f"{keeper.name}_stitched_layers.dat"), "w") as f:
                f.write(f"{'iteration':<12}{'layer':<8}{'press.[bar]':<16}{'flips':<8}{'in_final':<10}reason\n")
                for iteration, layer, reason in policy.events:
                    press = p_int[0] if layer == n else p_lay[layer]
                    f.write(f"{iteration:<12}{layer:<8}{press * 1e-6:<16.6e}{policy.flips[layer]:<8}"
                            f"{int(layer in policy.filled_now):<10}{reason}\n")
            log.warning(f"{keeper.name}: radiative layers stitched to convective: {stitched_final or 'none at the end'} "
                        f"(see {keeper.name}_stitched_layers.dat)")

        converged = not keeper.chelio_aborted
        with open(os.path.join(run_dir, f"{keeper.name}_coupling_convergence.dat"), "w") as f:
            f.write("1" if converged else "0")
        flippy = np.flatnonzero(policy.flips >= max(3, (policy.flip_limit or 100) // 10))
        info = {
            "converged": converged,
            "conv_iter": int(keeper.chelio_conv_iter),
            "t_setup_s": round(t_setup - t0, 2),
            "t_loops_s": round(t_loops - t_setup, 2),
            "t_total_s": round(time.time() - t0, 2),
            "T_surface": float(T_lay[n]),
            "T_min": float(T_lay.min()),
            "T_max": float(T_lay.max()),
            "n_convective_layers": int(conv_layer[:n].sum()),
            "surface_convective": bool(conv_layer[n]),
            "stitched_layers_final": stitched_final,
            "stitch_events": [list(e) for e in policy.events],
            "flipping_layers": {int(k): int(policy.flips[k]) for k in flippy},
            "vmr_provider": repr(keeper.chelio_vmr_provider) if keeper.chelio_vmr_provider is not None else None,
            "diagnostics": keeper.chelio_diag,
        }
        provider = keeper.chelio_vmr_provider
        if provider is not None and hasattr(provider, "stats"):
            info["vmr_provider_stats"] = provider.stats()
        info["T_profile"] = {"p_bar": p.tolist(), "T": T.tolist()}
        self._last_run_dir = run_dir
        return info


def write_session_info(run_dir, name, info):
    with open(os.path.join(run_dir, f"{name}_session.json"), "w") as f:
        json.dump(info, f, indent=1)
