# CHELIO

Coupling [HELIOS](https://github.com/exoclime/HELIOS) with [GGchem](https://github.com/pw31/GGchem) 

```
chelio
├─ README.md
├─ analyze
│  ├─ Tsurf_compare.ipynb
│  ├─ Tsurf_matrix.ipynb
│  ├─ analyze_TP+chemistry.ipynb
│  ├─ analyze_habitability.ipynb
│  ├─ comparisons.ipynb
│  ├─ contributions+opacities.ipynb
│  ├─ images
│  │  ├─ ...
│  ├─ opacity_contributions.ipynb
│  ├─ test_atmospheric_escape.ipynb
│  └─ test_redoxState.ipynb
├─ ggchem_inputs
│  ├─ abundances.in
│  ├─ param.in
│  ├─ param_test.in
│  └─ pt_helios.in
├─ helios_inputs
│  ├─ mixfile.dat
│  ├─ param.dat
│  ├─ param_io.dat
│  ├─ param_test.dat
│  ├─ species.dat
│  └─ species_test.dat
├─ multiple_runs.bash
├─ output
│  ├─ EqCond+Remove
│  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_Rayleigh_cross_sect.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_TOA_flux_eclipse.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_cloud_mixing_ratio.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_cloud_opacities.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_cloud_optdepth.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_cloud_scat_cross_sect.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_colmass_mu_cp_kappa_entropy.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_contribution.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_coupling_convergence.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_direct_beamflux.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_flux_ratio.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_g_0.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_integrated_flux.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_mean_extinct.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_opacities.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_optdepth.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_param.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_planck_cent.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_planck_int.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_run_info.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_spec_downflux.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_spec_upflux.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_started_convection.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_surf_albedo.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_-1.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_0.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_1.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_10.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_2.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_3.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_4.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_5.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_6.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_7.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_8.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_coupling_9.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_tp_cut.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_transmission.dat
│  │  │  ├─ Io_P0=1e6_Tint=150_CplusO=1e-2_CtoO=0.59_transweight.dat
│  │  │  ├─ Static_Conc_0.dat
│  │  │  ├─ Static_Conc_1.dat
│  │  │  ├─ Static_Conc_10.dat
│  │  │  ├─ Static_Conc_11.dat
│  │  │  ├─ Static_Conc_2.dat
│  │  │  ├─ Static_Conc_3.dat
│  │  │  ├─ Static_Conc_4.dat
│  │  │  ├─ Static_Conc_5.dat
│  │  │  ├─ Static_Conc_6.dat
│  │  │  ├─ Static_Conc_7.dat
│  │  │  ├─ Static_Conc_8.dat
│  │  │  ├─ Static_Conc_9.dat
│  │  │  ├─ abundances.in
│  │  │  ├─ escape.dat
│  │  │  ├─ param_ggchem.in
│  │  │  ├─ vertical_mix_0.dat
│  │  │  ├─ vertical_mix_1.dat
│  │  │  ├─ vertical_mix_10.dat
│  │  │  ├─ vertical_mix_11.dat
│  │  │  ├─ vertical_mix_2.dat
│  │  │  ├─ vertical_mix_3.dat
│  │  │  ├─ vertical_mix_4.dat
│  │  │  ├─ vertical_mix_5.dat
│  │  │  ├─ vertical_mix_6.dat
│  │  │  ├─ vertical_mix_7.dat
│  │  │  ├─ vertical_mix_8.dat
│  │  │  └─ vertical_mix_9.dat
│  │  ├─ ...
├─ pt_helios.in
├─ run_coupled.bash
└─ source
   ├─ calc_abundances.py
   ├─ calc_abundances_benchmark.py
   ├─ calc_escape.py
   ├─ convert_mixfile.py
   ├─ convert_tp.py
   ├─ create_pt.py
   └─ mark_bad_last_iters.py

```