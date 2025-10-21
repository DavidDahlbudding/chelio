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
out_file = 'test_pt_resx2.in'

# read and edit file line by line
with open(read_tp, 'r') as f:
    lines = f.readlines()
    if float(lines[1].split()[1]) == 1.001:
        convergence = False
    else:
        convergence = True
    
    pt = np.array([l.split() for l in lines[1:] if l.strip()], dtype=float)
    
    new_p = pt[:,0]
    new_p = 10**((np.log10(new_p[:-1]) + np.log10(new_p[1:])) / 2)
    new_t = np.interp(np.log10(new_p)[::-1], np.log10(pt[::-1,0]), pt[::-1,1])[::-1] # pressure has to monotonically increase as intput to interp
    
    n_lines_new = (len(lines)-2)*2 + 2
    new_pt = np.zeros((n_lines_new-1, 2))
    new_pt[::2,:] = pt
    new_pt[1::2,:] = np.column_stack((new_p, new_t))
    
    with open(out_file, 'w') as out:
        for i in range(n_lines_new):
            if i==0:
                out.writelines(lines[0])
            else:
                pt = new_pt[i-1,:]
                if convergence:
                    pt[1] = np.maximum(pt[1], Tmin)
                out.writelines(
                    "{:<24g}".format(pt[0])
                    + "{:<18g}\n".format(pt[1])
                )