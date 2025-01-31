import numpy as np
import argparse
import os
import warnings

print('Calculating abundances...')

# Calclate default (solar) C+O and C/O ratios
a_HCO = np.array([12, 8.46, 8.69]) # solar H, C, O from Asplund 2020

a_HCO = 10**(a_HCO-12) # convert
a_HCO = a_HCO / np.sum(a_HCO) # renormalize

CplusO_default = a_HCO[1] + a_HCO[2] # 7.775769e-04
CtoO_default = a_HCO[1] / a_HCO[2] # 0.588844

parser = argparse.ArgumentParser(description='Calculate abundances.')
parser.add_argument('--CplusO', type=float, default=CplusO_default, help='C+O/(C+O+H) ratio')
parser.add_argument('--CtoO', type=float, default=CtoO_default, help='C/O ratio')
parser.add_argument('--a_N', type=float, default=0.0, help='N abundance')

args = parser.parse_args()

CplusO = args.CplusO
CtoO = args.CtoO
a_N = args.a_N

if CplusO < 0:
    # e.g. CplusO = -1e-3 --> CplusO = 0.999
    CplusO = 1 - abs(CplusO)
elif CplusO == 1.0:
    warnings.warn('\nWarning: CplusO cannot be 1. Corrected to (1 - 1e-9).\nChoose negative value to set CplusO = 1 - abs(CplusO).')
    CplusO = 1 - 1e-9

a_H = (1 - a_N)
a_H = a_H * (1 - CplusO)
a_C = CtoO / (1 + CtoO) * (1 - a_N - a_H)
a_O = 1 - a_N - a_H - a_C

# consistency checks
rel_diff = 1e-6
assert np.abs(a_H + a_C + a_O + a_N - 1) < rel_diff, 'Sum of abundances is not 1'
assert np.abs((a_C + a_O)/(a_C + a_O + a_H) - CplusO) / CplusO < rel_diff, 'Final C+O abundance differs from input'
assert np.abs((a_C / a_O) - CtoO) / CtoO < rel_diff, 'Final C/O ratio differs from input'

x_H = 12.0
x_O = np.log10(a_O/a_H) + 12
x_C = np.log10(a_C/a_H) + 12
x_N = np.log10(a_N/a_H) + 12

print('H    ', x_H)
print('O    ', x_O)
print('C    ', x_C)
print('N    ', x_N)

# Save to file
filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ggchem_inputs/abundances.in'))

with open(filename, 'w') as f:
    f.write(f'H  {x_H:.5f}\n')
    f.write(f'O  {x_O:.5f}\n')
    f.write(f'C  {x_C:.5f}\n')
    f.write(f'N  {x_N:.5f}\n')