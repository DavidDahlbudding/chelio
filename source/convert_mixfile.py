import numpy as np

import time
import sys
import os

# import species database with args ['name', 'fc_name', 'weight']
# for name conversion and calculation of mean molecular weight
sys.path.append(os.path.join(os.environ['HELIOS_PATH'], 'source'))
from species_database import species_lib

print('Converting GGchem output to HELIOS input format ...')

# read in output file from second arg of command
if len(sys.argv) > 1:
    write_to = sys.argv[1]
else:
    write_to = os.path.abspath(os.path.join(os.path.dirname(__file__), '../helios_inputs/mixfile.dat'))

# read relevant species from helios_inputs/species.dat
species = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), '../helios_inputs/species.dat')), dtype=str, usecols=(0,))[1:]

# remove CIA (assumes that CIA species are already accounted for!)
species = np.array([s for s in species if s[:3] != 'CIA'])


# read GGchem output file
ggchem_output = os.path.join(os.environ['GGCHEM_PATH'], 'Static_Conc.dat')
header = np.loadtxt(ggchem_output, skiprows=2, max_rows=1, dtype=str)
dimension = np.genfromtxt(ggchem_output, dtype=int,  max_rows=1, skip_header=1)
data = np.loadtxt(ggchem_output, skiprows=3)

n_elem = dimension[0]
n_mol = dimension[1]
n_dust = dimension[2]
n_layers = dimension[3]

conversions = {'P(bar)': 'pgas', 'T(k)': 'Tg', 'n_<tot>(cm-3)': 'calculated_ntot', 'm(u)': 'calculated_mu', 'e-': 'el'}

# create header with correct (species) names
new_header = list(conversions.keys())
#new_header.extend([species_lib[s].fc_name for s in header[4:4+n_elem+n_mol] if s in species]) # fc = fastchem
new_header.extend([species_lib[s].name for s in header[4:4+n_elem+n_mol] if s in species])
new_header = np.array(new_header)

# convert data
new_data = np.zeros((n_layers, len(new_header)))

# Pressure (convert from cgs (dyn/cm^2) to bar)
new_data[:,0] = data[:,np.where(header == conversions[new_header[0]])[0][0]] * 1e-6

# Temperature
new_data[:,1] = data[:,np.where(header == conversions[new_header[1]])[0][0]]

# Calculate total number density
n_tot = 10**data[:,3:4+n_elem+n_mol]
n_tot = np.sum(n_tot, axis=1)
new_data[:,2] = n_tot

# Calculate mean molecular weight
mu = np.zeros(n_layers)
a_tot = np.zeros(n_layers)

for i,s in enumerate(header[3:4+n_elem+n_mol]):
    if s in species_lib.keys():
        a_mol = 10**data[:,3+i] # number density
        a_mol = a_mol / n_tot # fraction
        mu += a_mol * species_lib[s].weight
        a_tot += a_mol

    elif s == 'el':
        s = 'e-'
        a_mol = 10**data[:,3] # number density
        a_mol = a_mol / n_tot # fraction
        mu += a_mol * species_lib[s].weight
        a_tot += a_mol

    # save species fractions
    if s in species or s == 'e-':
        #new_data[:,np.where(new_header == species_lib[s].fc_name)[0][0]] = a_mol
        new_data[:,np.where(new_header == species_lib[s].name)[0][0]] = a_mol

if np.any(a_tot < 0.99):
    print("Warning: sum of considered species fractions is less than 1 in some layers!")
    time.sleep(0.5)

new_data[:,3] = mu

# nicely format header
header_string = []
for i in range(len(new_header)):
    header_string.append(new_header[i])
    n_spaces = 16 - len(new_header[i])
    header_string.append(n_spaces*' '+'\t')
header_string = ''.join(header_string[:-1])

# save to file
np.savetxt(write_to, new_data, header=header_string, fmt='%.10e', comments='', delimiter='\t')

