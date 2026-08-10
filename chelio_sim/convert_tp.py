import numpy as np
import sys
import os

# read in input file from second arg of command
if len(sys.argv) == 2:
    read_tp = sys.argv[1]
    Tmin = 20
elif len(sys.argv) == 3:
    read_tp = sys.argv[1]
    Tmin = float(sys.argv[2])
else:
    print('No input file provided. Exiting ...')
    sys.exit(1)

#out_file = os.path.join(os.environ['GGCHEM_PATH'], 'structures/pt_helios.in')
out_file = os.path.join(os.environ['GGCHEM_PATH'], 'structures/pt_helios.in')

# read and edit file line by line
with open(read_tp, 'r') as f:
    lines = f.readlines()
    if float(lines[1].split()[1]) == 1.001:
        convergence = False
    else:
        convergence = True
    with open(out_file, 'w') as out:
        for i, line in enumerate(lines):
            if i==0:
                out.writelines(line)
            else:
                pt = np.array(line.split(), dtype=float)
                if convergence:
                    pt[1] = np.maximum(pt[1], Tmin)
                out.writelines(
                    "{:<24g}".format(pt[0])
                    + "{:<18g}\n".format(pt[1])
                )