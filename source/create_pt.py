import numpy as np
import argparse
import os

print('Creating initial P-T-profile...')

parser = argparse.ArgumentParser(description='Create initial P-T-profile.')
parser.add_argument('--Teq', type=float, default=200, help='Equilibrium Temperature')
parser.add_argument('--Pmin', type=float, default=1e0, help='Minimum Pressure [1e-6 bar]')
parser.add_argument('--Pmax', type=float, default=1e6, help='Maximum Pressure [1e-6 bar]')

args = parser.parse_args()

Teq = args.Teq
Pmin = args.Pmin * 1e-6
Pmax = args.Pmax * 1e-6

nlayer = np.int32(np.ceil(10.5 * np.log10(Pmax / Pmin)) + 1)

P = np.logspace(np.log10(Pmax), np.log10(Pmin), nlayer)
T = np.ones_like(P) * Teq

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ggchem_inputs/pt_helios.in'))

with open(filename, 'w') as f:
    f.write(f'# P [bar], T [K]\n')
    for i in range(nlayer):
        f.write(f'{P[i]:.6e} {T[i]:.6e}\n')