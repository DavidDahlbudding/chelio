import numpy as np
import warnings
import os

# parameters
params = np.array(['P0', 'Tint', 'CplusO', 'CtoO'])

P0s = np.array([1e6, 1e7]) # surface pressure in dyn/cm^2
Tints = np.array([100, 150, 200, 250, 300, 350]) # internal temperature in K

CplusOs = np.array([1e-6, 3.16e-5, 1e-3, 3.16e-2, 1e0])
CtoOs = np.array([0.1, 0.59, 1.2])

folder = '../output/EqChem/'


def format_e_nums(num):
    num = f'{num:.2e}'.replace('0', '').replace('.e', 'e').replace('+', '')
    if num[-1] == 'e':
        num = num + '0'
    return num

def format_CtoO_float(f):
    if f == int(f):  # Check if it's a whole number
        return f"{f:.1f}"  # Format as integer if whole
    else:
        return f"{f:.10g}" # Use g format with sufficient precision.

def built_name(P0, Tint, CplusO, CtoO):
    return f'Earth_P0={format_e_nums(P0)}_Tint={Tint}_NoCond_CplusO={format_e_nums(CplusO)}_CtoO={format_CtoO_float(CtoO)}'
    
def extract_data(i_P0=0, i_Tint=0, i_CplusO=0, i_CtoO=0):
        
    inds = []
    PTs = []

    name = built_name(P0s[i_P0], Tints[i_Tint], CplusOs[i_CplusO], CtoOs[i_CtoO])

    path = folder + name + "/Static_Conc_{var}.dat"
    j = 0

    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                # if bad file exists
                if os.path.isfile(path.format(var=f'{j}_bad')):
                    # rename file
                    os.rename(path.format(var=f'{j}_bad'), path.format(var=j))
                d = np.loadtxt(path.format(var=j), skiprows=3)
                PTs.append(np.array([d[:,2]*1e-6, d[:,0]]).T) # convert pressure from dyn/cm^2 to bar
                inds.append(j)
                j += 1
            except (FileNotFoundError, UserWarning) as warn:
                if warn.__class__ == UserWarning:
                    print(f'!GGchem did not converge for {name}!')
                break

    return name, np.array(inds), np.array(PTs)

for i in range(len(P0s)):
    for j in range(len(Tints)):
        for k in range(len(CplusOs)):
            for l in range(len(CtoOs)):
                name, inds, PTs = extract_data(i, j, k, l)

                # calc mean squared difference between consecutive points
                diff = np.sqrt((np.abs(PTs[1:,:,1] - PTs[:-1,:,1])**2).mean(axis=1))

                for m in np.arange(len(diff)-1, 0, -1):
                    if diff[m] > 1.1*diff[m-1] and diff[m] > 1e0:
                        print(f'Bad last iterations for {name} at index {inds[m]}')
                        print('Differences: ', diff)
                        
                        # Rename files
                        #os.rename(folder + name + f'/Static_Conc_{m}.dat', folder + name + f'/Static_Conc_{m}_bad.dat')
                    else:
                        break
