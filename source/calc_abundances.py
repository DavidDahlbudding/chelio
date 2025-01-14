import numpy as np
import argparse
import os

print('Calculating abundances...')

parser = argparse.ArgumentParser(description='Calculate abundances.')
parser.add_argument('--OminH', type=float, default=0.0, help='OminH value')
parser.add_argument('--a_C', type=float, default=0.15, help='a_C value')
parser.add_argument('--a_N', type=float, default=1e-3, help='a_N value')

args = parser.parse_args()

OminH = args.OminH
a_C = args.a_C
a_N = args.a_N

a_C = a_C * (1 - a_N)
a_OplusH = 1 - a_C - a_N

a_O = (OminH*0.5 + 0.5) * a_OplusH
a_H = a_OplusH - a_O

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