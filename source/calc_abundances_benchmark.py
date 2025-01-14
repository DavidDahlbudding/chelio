#!/usr/bin/env python3

import math
import os

def read_solar_abundances(filename):
    """
    Reads the original abundance file and returns a dictionary:
       abundances[element_symbol] = solar_number_fraction
    The 'Solar' column is taken as index 6 (0-based).
    
    Lines look like:
        1 hydrogen H 1.00794  2.887E-02  6.653E-01  9.271E-01  3.578E-01
        |    |     |    |         |          |          |          |
        0    1     2    3         4          5          6 (Solar)  7
    """
    solar_col_index = 6
    relevant_species = "H He C N O Na Mg Al Si P S Cl K Ca Ti V Fe Ni el".split()
    
    abund_solar = {}
    with open(filename, 'r') as f:
        for line in f:
            # Skip blank lines or lines that don't look like data
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.split()
            # We expect at least 7 columns
            if len(parts) < 7:
                continue

            # Skip elements not in the list
            if parts[2] not in relevant_species:
                continue
            
            # Try to parse:
            #   parts[0] -> atomic number (string)
            #   parts[1] -> element name (string)
            #   parts[2] -> element symbol (string)
            #   parts[3] -> atomic mass
            #   parts[4] -> EarthCrust
            #   parts[5] -> Ocean
            #   parts[6] -> Solar
            #   parts[7] -> Meteorites (if present)
            try:
                symbol = parts[2]
                val_solar = float(parts[solar_col_index])
            except ValueError:
                continue  # skip line if parsing fails
            
            # Store
            abund_solar[symbol] = val_solar
    
    return abund_solar

def compute_scaled_abundances(
    abund_sun, 
    delta_fe_h = 1.0,   # desired [Fe/H]
    c_o_target = 0.8    # desired C/O
):
    """
    Given a dictionary of solar abundances (by number), produce a new 
    composition with:
       [Fe/H] = delta_fe_h
       C/O    = c_o_target
    We keep H, He as solar; scale all metals by factor f=10^(Delta), 
    then readjust C and O to get the final ratio c_o_target, 
    and finally renormalize.
    
    Returns a new dictionary of final number fractions (that sum to 1).
    """
    # 1) Extract solar values for H, He, C, O, Fe
    A_H_sun  = abund_sun.get("H",  0.0)
    A_He_sun = abund_sun.get("He", 0.0)
    A_C_sun  = abund_sun.get("C",  0.0)
    A_O_sun  = abund_sun.get("O",  0.0)
    A_Fe_sun = abund_sun.get("Fe", 0.0)
    
    # 2) Define scale factor f from [Fe/H] = Delta => f = 10^Delta
    f = 10.0**delta_fe_h
    
    # 3) Scale all metals except C and O
    new_abund = {}
    for elem, val_sun in abund_sun.items():
        if elem in ("H","He","C","O"):
            # We'll handle these specially
            continue
        # metals heavier than He => scale
        new_abund[elem] = f * val_sun
    
    # 4) "Pre-adjust" C and O by the same factor f
    C_0 = f * A_C_sun
    O_0 = f * A_O_sun
    
    # 5) Adjust C and O to achieve desired C/O, but keep total (C+O) the same
    sumCO = C_0 + O_0
    if sumCO > 0.0:
        O_final = sumCO / (1.0 + c_o_target)
        C_final = c_o_target * O_final
    else:
        # Edge case if sumCO=0 (maybe no C or O in solar data),
        # then we can't fix ratio. We'll just keep them 0.
        O_final = 0.0
        C_final = 0.0
    
    # 6) Build final composition BEFORE normalization
    # H, He are left as solar (per your note)
    A_H_new  = A_H_sun
    A_He_new = A_He_sun
    
    # metals:
    new_abund["C"] = C_final
    new_abund["O"] = O_final
    # H, He will be stored after we renormalize
    
    # 7) Compute sum & renormalize
    sum_metals = sum(new_abund.values())
    sum_all = A_H_new + A_He_new + sum_metals
    
    # renormalization factor
    if sum_all <= 0:
        raise ValueError("Sum of new abundances is zero or negative, check data.")
    renorm = 1.0 / sum_all
    
    # final dictionary of number fractions
    final_abund = {}
    # H, He
    final_abund["H"]  = A_H_new  * renorm
    final_abund["He"] = A_He_new * renorm
    # metals
    for elem, val in new_abund.items():
        final_abund[elem] = val * renorm
    
    return final_abund

def main():
    # -----------------------------------------------------------
    # 1) User inputs: file, desired metallicity, and C/O ratio
    # -----------------------------------------------------------
    input_file  = os.path.join(os.environ['GGCHEM_PATH'], 'data/Abundances.dat')  # Adjust path/name as needed
    out_file    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ggchem_inputs/abundances.in'))
    
    desired_fe_h = 1.0   # i.e. [Fe/H] = +1
    desired_co   = 0.8   # final C/O ratio
    
    # -----------------------------------------------------------
    # 2) Read solar abundances from the input file
    # -----------------------------------------------------------
    abund_sun = read_solar_abundances(input_file)
    
    # Quick check that we have something for H, Fe, etc.
    if "H" not in abund_sun or "Fe" not in abund_sun:
        raise ValueError("Could not find H or Fe in the solar abundance data!")
    
    # -----------------------------------------------------------
    # 3) Compute new composition
    # -----------------------------------------------------------
    final_abund = compute_scaled_abundances(
        abund_sun,
        delta_fe_h = desired_fe_h,
        c_o_target = desired_co
    )
    
    # -----------------------------------------------------------
    # 4) Sanity checks: [Fe/H] and C/O
    # -----------------------------------------------------------
    # solar ratio Fe/H
    fe_h_sun = abund_sun["Fe"] / abund_sun["H"] if abund_sun["H"]>0 else 0
    # new ratio Fe/H
    fe_h_new = final_abund.get("Fe", 0.0) / final_abund["H"]
    # computed [Fe/H]
    if fe_h_sun>0:
        final_fe_h = math.log10(fe_h_new / fe_h_sun)
    else:
        final_fe_h = float('nan')
    
    c_o_new = (final_abund.get("C",0.0) / final_abund["O"]) if final_abund.get("O",0.0)>0 else 0
    print(f"Sanity check: [Fe/H]_final = {final_fe_h:.3f}  (target was {desired_fe_h})")
    print(f"Sanity check: (C/O)_final  = {c_o_new:.3f}    (target was {desired_co})")
    
    # -----------------------------------------------------------
    # 5) Output in log scale with H = 12
    #    x_i = log10( n_i / n_H ) + 12
    # -----------------------------------------------------------
    # By definition, log-epsilon(H) = 12
    # So for element i, log-epsilon(i) = log10( final_abund[i] / final_abund[H] ) + 12
    # We'll sort elements by atomic number if we want, but we only have symbols. 
    # For simplicity, we can just sort by symbol or output in alphabetical order.
    
    # Create a sorted list of elements (with H first if you like).
    # We'll ensure "H" is at the front, everything else alphabetical:
    sorted_elems = sorted(final_abund.keys(), key=lambda e: (e != "H", e))
    
    # Write to file
    with open(out_file, 'w') as out:
        for elem in sorted_elems:
            if elem == "H":
                # By definition
                logeps = 12.0
            else:
                ratio = final_abund[elem] / final_abund["H"]
                if ratio <= 0:
                    # Could happen if final_abund[elem] == 0
                    logeps = -999.0
                else:
                    logeps = math.log10(ratio) + 12.0
            out.write(f"{elem} {logeps:.16f}\n")
    
    print(f"\nWrote output to {out_file}")
    print("First few lines:")
    with open(out_file, 'r') as fcheck:
        for i, line in enumerate(fcheck):
            print(line.strip())
            if i>5:
                break

if __name__ == "__main__":
    main()
