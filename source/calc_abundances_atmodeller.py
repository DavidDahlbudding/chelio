#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
import warnings

helios_path = os.environ['HELIOS_PATH']
if helios_path not in sys.path:
    sys.path.append(helios_path)
from source.species_database import species_lib

import logging
from atmodeller import (
    InteriorAtmosphere,
    Planet,
    Species,
    SpeciesCollection,
    debug_logger,
    earth_oceans_to_hydrogen_mass
)
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

logger = debug_logger()
logger.setLevel(logging.INFO)

m_earth = 5.972e24 # kg
r_earth = 6.371e6 # m

print('Calculating abundances...')

parser = argparse.ArgumentParser(description='Calculate abundances.')

# Planet parameters
parser.add_argument('--M_P', type=float, default=m_earth, help='Planet mass')
parser.add_argument('--R_P', type=float, default=r_earth, help='Planet radius')
parser.add_argument('--core_frac', type=float, default=0.295334691460966, help='Core mass fraction')
parser.add_argument('--melt_frac', type=float, default=1.0, help='Melt fraction')
parser.add_argument('--T_surf', type=float, default=2000.0, help='Surface temperature')

# Composition parameters
parser.add_argument('--H_ocean', type=float, default=1.0, help='H ocean abundance')
parser.add_argument('--CtoH', type=float, default=1.0, help='C/H mass ratio')
parser.add_argument('--NtoC', type=float, default=0.1, help='N/C mass ratio')
parser.add_argument('--fO2', type=float, default=0.0, help='Oxygen fugacity fO2 [delta IW]')
parser.add_argument('--StoC', type=float, default=None, help='S/C mass ratio')
parser.add_argument('--CltoC', type=float, default=None, help='Cl/C mass ratio')

parser.add_argument('--output_dir', type=str, default='ggchem', help='Output (ggchem or helios)')

args = parser.parse_args()

M_P = args.M_P
R_P = args.R_P
core_frac = args.core_frac
melt_frac = args.melt_frac
T_surf = args.T_surf

H_ocean = args.H_ocean
CtoH = args.CtoH
NtoC = args.NtoC
fO2 = args.fO2
StoC = args.StoC
CltoC = args.CltoC

output_dir = args.output_dir
if output_dir not in ['ggchem', 'helios']:
    raise ValueError('Output must be either ggchem or helios')

# Setup Planet
planet = Planet(
    planet_mass=M_P,
    mantle_melt_fraction=melt_frac,
    surface_radius=R_P,
    surface_temperature=T_surf,
    core_mass_fraction=core_frac
)

available_species = SpeciesCollection.available_species()
solubility_models = get_solubility_models()

# Default model only includes C-H-N-O
H2O_g: Species = Species.create_gas("H2O", solubility=solubility_models["H2O_basalt_dixon95"])
H2_g: Species = Species.create_gas("H2", solubility=solubility_models["H2_basalt_hirschmann12"])
CO_g: Species = Species.create_gas("CO", solubility=solubility_models["CO_basalt_yoshioka19"])
CO2_g: Species = Species.create_gas("CO2", solubility=solubility_models["CO2_basalt_dixon95"])
CH4_g: Species = Species.create_gas("CH4", solubility=solubility_models["CH4_basalt_ardia13"])
N2_g: Species = Species.create_gas("N2", solubility=solubility_models["N2_basalt_libourel03"])
O2_g: Species = Species.create_gas("O2")
NH3_g: Species = Species.create_gas("NH3")

# condensed graphite
C_s = Species.create_condensed("C")

species = (H2O_g, H2_g, CO_g, CO2_g, CH4_g, N2_g, O2_g, NH3_g, C_s)

h_kg = earth_oceans_to_hydrogen_mass(H_ocean)
c_kg = CtoH * h_kg
n_kg = NtoC * c_kg
mass_constraints = {
    "H": h_kg,
    "C": c_kg,
    "N": n_kg
}
fugacity_constraints = {O2_g.name: IronWustiteBuffer(fO2)}

if StoC is not None:
    S2_g: Species = Species.create_gas("S2", solubility=solubility_models["S2_basalt_boulliung23"])
    SO2_g: Species = Species.create_gas("SO2")
    H2S_g: Species = Species.create_gas("H2S")
    SO_g: Species = Species.create_gas("SO")

    species += (S2_g, SO2_g, H2S_g, SO_g)
    mass_constraints["S"] = StoC * c_kg

if CltoC is not None:
    Cl2_g: Species = Species.create_gas("Cl2", solubility=solubility_models["Cl2_basalt_thomas21"])
    ClH_g: Species = Species.create_gas("ClH")
    
    species += (Cl2_g, ClH_g)
    mass_constraints["Cl"] = CltoC * c_kg

species = SpeciesCollection(species)
interior_atmosphere = InteriorAtmosphere(species)

# Initial solution guess number density (molecules/m^3)
initial_log_number_density = 50 * np.ones(len(species))

interior_atmosphere.solve(
    planet=planet,
    initial_log_number_density=initial_log_number_density,
    mass_constraints=mass_constraints,
    fugacity_constraints=fugacity_constraints,
)

output = interior_atmosphere.output
output_dict = output.asdict()


print('Saving results...')

if output_dir == 'ggchem':
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ggchem_inputs/abundances.in'))

    x_H = (output_dict['element_H']['atmosphere_moles']/output_dict['atmosphere']['element_moles'])[0]

    with open(filename, 'w') as f:
        for e in list(mass_constraints.keys()) + ['O']:
            x_e = (output_dict[f'element_{e}']['atmosphere_moles']/output_dict['atmosphere']['element_moles'])[0]
            x_e = np.log10(x_e/x_H) + 12
            print(f'{e} {x_e:.5f}')
            f.write(f'{e} {x_e:.5f}\n')


all_helios_species = os.path.abspath(os.path.join(os.path.dirname(__file__), '../helios_inputs/all_species.dat'))

translate_dict = { # translates species names from atmodeller to helios
    'H3N': 'NH3',
    'O2S': 'SO2',
    'OS': 'SO',
}

outgassed_species_vmrs = {}
for key in output_dict.keys():
    if key.endswith('_g'):
        vmr = output_dict[key]['volume_mixing_ratio'][0]
        if key[:-2] in translate_dict.keys():
            outgassed_species_vmrs[translate_dict[key[:-2]]] = vmr
        else:
            outgassed_species_vmrs[key[:-2]] = vmr

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../helios_inputs/species.dat'))

with open(filename, 'w') as f:
    with open(all_helios_species, 'r') as f_all:
        for i, line in enumerate(f_all.readlines()):
            if i == 0:
                print(line)
                f.write(line+'\n')
            elif line != '\n':
                
                if line.split()[0][:3] == 'CIA':
                    pair = species_lib[line.split()[0]].fc_name.replace('1', '').split('&')
                    if pair[0] in outgassed_species_vmrs.keys() and pair[1] in outgassed_species_vmrs.keys():
                        if output_dir == 'helios':
                            line = line.replace('file', f'{outgassed_species_vmrs[pair[0]]:.5e}&{outgassed_species_vmrs[pair[1]]:.5e}')
                    else:
                        continue
                elif line.split()[0] in outgassed_species_vmrs.keys():
                    if output_dir == 'helios':
                        line = line.replace('file', f'{outgassed_species_vmrs[line.split()[0]]:.5e}')
                else:
                    continue
                
                print(line)
                f.write(line+'\n')
                    

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../helios_inputs/P_BOA.dat'))

with open(filename, 'w') as f:
    BOA_P = output_dict['atmosphere']['pressure'][0]*1e6 # bar to dyn/cm^2
    f.write(f'{BOA_P:.5e}')

