# calc_escape.py
import argparse
import os
import traceback
import numpy as np
from scipy.interpolate import interp1d

# --- Constants (cgs units) ---
G = 6.674e-8  # cm^3 g^-1 s^-2
kB = 1.381e-16 # erg K^-1
avogadro = 6.022e23 # mol^-1
m_H = 1.66e-24 # g
m_Earth = 5.972e27 # g
r_Earth = 6.371e8 # cm
r_Io = 1.821e8 # cm
m_Io = 8.9319e25 # g
kin_dia_H2 = 289e-10 # cm

planet_properties = {
    "Earth": {"mass": m_Earth, "radius": r_Earth},
    "Io": {"mass": m_Io, "radius": r_Io},
    # Add more planets/moons here if needed
}

# --- Escape Functions from atmospheric_escape.ipynb ---
def maxTexo(m_P, r_P, mu):
    return mu * m_H * G * m_P / (36 * kB * r_P)

def radius_profile(P, r0, P0, T, mu, m_P):
    # return r0 + (kB * T) / (mu * m_H * g) * np.log(P0 / P)
    return (1/r0 - (kB * T) / (G * m_P * mu) * np.log(P0 / P))**(-1)

def extend_profile(r, P, T, Pmin, mu, m_P, n_tots):
    new_P = np.logspace(np.log10(P[-1]), np.log10(Pmin), int(np.log10(P[-1]/Pmin)*10+1))
    gravity = G * m_P / r[0]**2
    new_r = radius_profile(new_P, r[-1], P[-1], T[-1], mu[-1], gravity)
    new_ntots = new_P / (kB * T[-1])
    return np.concatenate((r, new_r)), np.concatenate((P, new_P)), np.concatenate((T, np.full_like(new_P, T[-1]))), np.concatenate((mu, np.full_like(new_P, mu[-1]))), np.concatenate((n_tots, new_ntots))

def r_exobase(press_dyn_cm2, m_P, mu, kin_dia): # pressure in dyn/cm^2
    r_c = G / (np.sqrt(2) * np.pi)
    r_c *= mu * m_H * m_P / (kin_dia**2 * press_dyn_cm2)
    r_c = np.sqrt(r_c)
    return r_c

def get_mfp(press, T, kin_dia):
    # calculate the mean free path
    return (kB * T) / (np.sqrt(2) * np.pi * kin_dia**2 * press)

def get_scale_height(r, T, m_P, mu):
    # calculate the scale height
    return kB * T * r**2 / (m_P * G * mu * m_H)

def get_sound_speed(T, mu):
    # calculate the sound speed
    return np.sqrt(kB * T / (mu * m_H))

def get_bondi_radius(m_P, c_s):
    # calculate the Bondi radius
    return G * m_P / (2 * c_s**2)

def escape_rate(r_c, P_c, T_c, m_P, mu, B=0.65):
    lambda_c = G * m_P * mu * m_H / (kB * T_c * r_c)
    n_c = P_c / (kB * T_c)
    phi_jeans = n_c / (2 * np.sqrt(np.pi)) * B
    phi_jeans *= np.sqrt(2 * kB * T_c / (mu * m_H))
    phi_jeans *= (1 + lambda_c) * np.exp(-lambda_c)
    return lambda_c, phi_jeans

def find_last_iteration(folder_path, file_prefix):
    i = 0
    last_i = 0
    while True:
        file_path = os.path.join(folder_path, f"{file_prefix}{i}.dat")
        if not os.path.exists(file_path):
            break
        last_i = i
        i += 1
    return last_i

def calculate_escape_parameters(folder_path, folder_name):
    # 1. Data Extraction
    i_max_static = find_last_iteration(os.path.join(folder_path, folder_name), "Static_Conc_")
    i_max_vertical_mix = find_last_iteration(os.path.join(folder_path, folder_name), "vertical_mix_")
    static_data_path = os.path.join(folder_path, folder_name, f"Static_Conc_{i_max_static}.dat")
    vertical_mix_path = os.path.join(folder_path, folder_name, f"vertical_mix_{i_max_vertical_mix}.dat")
    tp_data_path = os.path.join(folder_path, folder_name, f"{folder_name}_tp.dat")
    flux_data_path = os.path.join(folder_path, folder_name, f"{folder_name}_integrated_flux.dat")

    # Load data using np.loadtxt, skip header and dimension lines (4 rows)
    try:
        static_data = np.loadtxt(static_data_path, skiprows=3) # altitude (cm) in column 0, pressure (dyn/cm^2) in column 2, Temp (K) in column 2, nHtot in column 1
        vertical_mix_mu = np.loadtxt(vertical_mix_path, skiprows=1) # mu in column 3, skip 2 header rows
        tp_data_alt = np.loadtxt(tp_data_path, skiprows=2, usecols=3) # altitude (cm) in column 3
        flux_data = np.loadtxt(flux_data_path, skiprows=3, usecols=4) # F_net in column 4
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data file: {e}")
    except ValueError as e: # Catch errors during data loading (e.g., wrong skiprows)
        raise ValueError(f"Error loading data from file: {e}. Check file format and skiprows settings.")

    altitudes = tp_data_alt[:] # cm
    pressures_dyn_cm2 = static_data[:, 2] # dyn/cm^2 (column index 2)
    temperatures = static_data[:, 0] # K (column index 2)
    n_tots = vertical_mix_mu[:, 2] # nHtot values
    mu = vertical_mix_mu[:, 3] # mu values
    F_net = flux_data[:] # erg s^-1 cm^-2

    # 1b. Planetary Properties from folder name
    planet_name = folder_name.split('_')[0] # Assumes planet name is before the first "_"
    if planet_name in planet_properties:
        m_P = planet_properties[planet_name]["mass"]
        r_P = planet_properties[planet_name]["radius"]
    else:
        print(f"Warning: Planet name '{planet_name}' not found in planet properties. Using Earth properties.")
        m_P = m_Earth
        r_P = r_Earth

    radius = altitudes + r_P # cm
    L_cool = F_net[-1] * 4 * np.pi * r_P**2 # erg s^-1
    
    # Calculate atmospheric mass based on surface pressure and planetary radius
    grav = G * m_P / r_P**2 # cm/s^2
    M_atmo = pressures_dyn_cm2[0] * 4 * np.pi * r_P**2 / grav # g

    # 2. Boundary Calculation (Exobase and Bondi)
    mfp = get_mfp(pressures_dyn_cm2, temperatures, kin_dia_H2)
    scale_height = get_scale_height(radius, temperatures, m_P, mu)
    c_s_profile = get_sound_speed(temperatures, mu)
    r_bondi_profile = get_bondi_radius(m_P, c_s_profile)
    r_c_values = r_exobase(pressures_dyn_cm2, m_P, mu, kin_dia_H2) # exobase radius profile

    while (mfp[-1] < scale_height[-1]) and (radius[-1] < r_bondi_profile[-1]):
        # approximate P at the next boundary to extend the profile
        # a lower pressure is needed, so a smaller value is chosen
        if mfp[-1] < scale_height[-1]:
            # approximate P at exobase if not yet reached
            approx_P_exo = r_exobase(pressures_dyn_cm2[-1], m_P, mu[-1], kin_dia_H2)**2 * pressures_dyn_cm2[-1] / (radius[-1]**2)
        else:
            approx_P_exo = pressures_dyn_cm2[-1]

        if radius[-1] < r_bondi_profile[-1]:
            # approximate P at bondi radius if not yet reached
            approx_P_bondi = pressures_dyn_cm2[-1] * np.exp(-(r_bondi_profile[-1] - radius[-1]) / scale_height[-1])
        else:
            approx_P_bondi = pressures_dyn_cm2[-1]
        
        approx_P = approx_P_exo #min(approx_P_exo, approx_P_bondi)
        approx_P = 10**np.floor(np.log10(approx_P))
        print(f"Extending profile to P = {approx_P} dyn/cm^2 from {pressures_dyn_cm2[-1]} dyn/cm^2")
        
        # extend profiles to new P
        radius, pressures_dyn_cm2, temperatures, mu, n_tots = extend_profile(radius, pressures_dyn_cm2, temperatures, approx_P, mu, m_P, n_tots)
        altitudes = radius - r_P
        
        # recalculate profiles
        mfp = get_mfp(pressures_dyn_cm2, temperatures, kin_dia_H2)
        scale_height = get_scale_height(radius, temperatures, m_P, mu)
        c_s_profile = get_sound_speed(temperatures, mu)
        r_bondi_profile = get_bondi_radius(m_P, c_s_profile)
        r_c_values = r_exobase(pressures_dyn_cm2, m_P, mu, kin_dia_H2)

    # Interpolate functions
    f_P = interp1d(radius, np.log10(pressures_dyn_cm2), bounds_error=False, fill_value="extrapolate")
    f_T = interp1d(radius, temperatures, bounds_error=False, fill_value="extrapolate")
    f_n = interp1d(radius, np.log10(n_tots), bounds_error=False, fill_value="extrapolate")
    f_mu = interp1d(radius, mu, bounds_error=False, fill_value="extrapolate")
    f_cs = interp1d(radius, c_s_profile, bounds_error=False, fill_value="extrapolate")
    f_rbondi = interp1d(radius, r_bondi_profile, bounds_error=False, fill_value="extrapolate")
    f_rc = interp1d(radius, r_c_values, bounds_error=False, fill_value="extrapolate")
    f_mfp = interp1d(radius, mfp, bounds_error=False, fill_value="extrapolate")
    f_sh = interp1d(radius, scale_height, bounds_error=False, fill_value="extrapolate")
    
    # Find exobase radius (mfp = scale_height)
    i_exo = np.sum(mfp < scale_height)
    if i_exo == len(mfp): # exobase is outside the profile
        r_exo = np.nan
    else:
        dsh = scale_height[i_exo] - scale_height[i_exo-1]
        dmfp = mfp[i_exo] - mfp[i_exo-1]
        dr = radius[i_exo] - radius[i_exo-1]
        r_exo = radius[i_exo-1] + (scale_height[i_exo-1] - mfp[i_exo-1]) * dr / (dmfp - dsh)

    # Find Bondi radius level (r = r_bondi)
    i_bondi = np.sum(radius < r_bondi_profile)
    if i_bondi == len(radius): # bondi radius is outside the profile
        r_bondi_level = np.nan
    else:
        drbondi = r_bondi_profile[i_bondi] - r_bondi_profile[i_bondi-1]
        dr = radius[i_bondi] - radius[i_bondi-1]
        r_bondi_level = radius[i_bondi-1] + (r_bondi_profile[i_bondi-1] - radius[i_bondi-1]) * dr / (dr - drbondi)

    #print(f"Exobase altitude: {r_exo:.2e} cm")
    #print(f"Exobase radius: {f_rc(r_exo):.2e} cm")

    # 3. Determine Escape Regime and Calculate Rates
    escape_regime = ""
    escape_time_atmo_yrs = np.nan
    
    # Prioritize Bondi-dominated escape if its radius is smaller than the exobase
    if not np.isnan(r_bondi_level) and (np.isnan(r_exo) or r_bondi_level < r_exo):
        # Bondi radius is reached first, calculate hydrodynamic escape
        
        # Interpolate properties at the Bondi level
        P_B = 10**f_P(r_bondi_level)
        T_B = f_T(r_bondi_level)
        n_B = 10**f_n(r_bondi_level)
        mu_B = f_mu(r_bondi_level)
        c_s_B = f_cs(r_bondi_level)
        
        # Check hydrodynamic criterion as a warning
        length_scale_B = 1 / (np.sqrt(2) * np.pi * kin_dia_H2**2 * n_B)
        if length_scale_B > r_bondi_level / 3:
            print(f"Warning for {folder_name}: Hydrodynamic criterion not met at Bondi level, but proceeding as requested.")
            print(f"  Length Scale: {length_scale_B:.2e} cm, r_bondi/3: {(r_bondi_level/3):.2e} cm")

        # Calculate mass loss rates
        rho_B = n_B * mu_B * m_H
        Mdot_hydro = 4 * np.pi * r_bondi_level**2 * rho_B * c_s_B  # g/s
        Mdot_energy = L_cool / (G * m_P / r_P)  # g/s, using r_P for potential energy

        if Mdot_hydro < Mdot_energy:
            escape_regime = "Hydrodynamic"
            Mdot_total = Mdot_hydro
        else:
            escape_regime = "Energy-limited"
            Mdot_total = Mdot_energy

        # Calculate atmospheric lifetime
        if Mdot_total > 0:
            escape_time_s = M_atmo / Mdot_total # s
            escape_time_atmo_yrs = escape_time_s / (60*60*24*365.25) # years
        else:
            escape_time_atmo_yrs = np.inf


    elif not np.isnan(r_exo):
        # Exobase is reached first (or Bondi level is not found), calculate Jeans escape
        escape_regime = "Jeans"
        
        # Get properties at exobase
        r_c = f_rc(r_exo)
        P_c = 10**f_P(r_exo)
        T_c = f_T(r_exo)
        n_c = 10**f_n(r_exo)
        mu_c = f_mu(r_exo)

        # Thermal escape condition
        T_max_exo = maxTexo(m_P, r_P, mu_c)
        thermal_escape_condition = T_c > T_max_exo

        # Jeans escape rate
        lambda_c, phi_jeans = escape_rate(r_c, P_c, T_c, m_P, mu_c) # particles cm^-2 s^-1
        Mdot_jeans = phi_jeans * (mu_c * m_H) * (4 * np.pi * r_c**2) # g/s
        
        # Calculate atmospheric lifetime
        if Mdot_jeans > 0:
            escape_time_s = M_atmo / Mdot_jeans # s
            escape_time_atmo_yrs = escape_time_s / (60*60*24*365.25) # years
        else:
            escape_time_atmo_yrs = np.inf


    else:
        # Neither boundary could be determined
        escape_regime = "Undetermined"
        print(f"Warning for {folder_name}: Could not determine escape regime.")


    # 4. Save Results
    output_path = os.path.join(folder_path, folder_name, "escape.dat")
    results = {"escape_regime": escape_regime, "escape_time_atmo_yrs": escape_time_atmo_yrs}

    with open(output_path, 'w') as f:
        f.write(f"Planet Name: {planet_name}\n")
        f.write(f"Escape Regime: {escape_regime}\n")
        f.write(f"Escape Timescale of entire Atmosphere [years]: {escape_time_atmo_yrs:.2e}\n")
        f.write("-" * 20 + "\n")

        if "Hydrodynamic" in escape_regime or "Energy-limited" in escape_regime:
            results.update({
                "r_bondi_level": r_bondi_level, "P_B": P_B, "T_B": T_B, "n_B": n_B,
                "Mdot_total": Mdot_total, "Mdot_hydro": Mdot_hydro, "Mdot_energy": Mdot_energy
            })
            f.write(f"Bondi Level Radius [cm]: {r_bondi_level:.4e}\n")
            f.write(f"Bondi Level Pressure [dyn/cm^2]: {P_B:.4e}\n")
            f.write(f"Bondi Level Temperature [K]: {T_B:.2f}\n")
            f.write(f"Bondi Level Number Density [cm^-3]: {n_B:.4e}\n")
            f.write(f"Total Mass Loss Rate [g/s]: {Mdot_total:.4e}\n")
            f.write(f"  (Hydrodynamic Mdot: {Mdot_hydro:.4e} g/s)\n")
            f.write(f"  (Energy-limited Mdot: {Mdot_energy:.4e} g/s)\n")

        elif escape_regime == "Jeans":
            results.update({
                "r_exo": r_exo, "r_c": r_c, "P_c": P_c, "T_c": T_c, "n_c": n_c,
                "lambda_c": lambda_c, "phi_jeans": phi_jeans, "thermal_escape_condition": thermal_escape_condition
            })
            f.write(f"Exobase Altitude [cm]: {r_exo:.4e}\n")
            f.write(f"Exobase Radius (r_c) [cm]: {r_c:.4e}\n")
            f.write(f"Exobase Pressure (P_c) [dyn/cm^2]: {P_c:.4e}\n")
            f.write(f"Exobase Temperature (T_c) [K]: {T_c:.2f}\n")
            f.write(f"Exobase Number Density (n_c) [cm^-3]: {n_c:.4e}\n")
            f.write(f"Thermal Escape Condition Met: {thermal_escape_condition}\n")
            f.write(f"Jeans Escape Parameter (lambda_c): {lambda_c:.4e}\n")
            f.write(f"Jeans Escape Rate [particles cm^-2 s^-1]: {phi_jeans:.4e}\n")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate atmospheric escape parameters.")
    parser.add_argument("folder_path", help="Path to the main folder containing subfolders.")
    args = parser.parse_args()
    main_folder_path = args.folder_path

    if not os.path.isdir(main_folder_path):
        print(f"Error: Folder path '{main_folder_path}' is not a valid directory.")
        exit(1)

    processed_folders = 0
    names = []
    summary_results = []

    for folder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(subfolder_path): # Check if it's a directory
            try:
                print(f"Processing folder: {folder_name}")
                escape_results = calculate_escape_parameters(main_folder_path, folder_name)
                names.append(folder_name)
                summary_results.append(escape_results)
                print(f"Escape parameters saved to {folder_name}/escape.dat")
                processed_folders += 1
            except FileNotFoundError as e:
                print(f"Warning: Could not process folder {folder_name}: {e}")
            except ValueError as e:
                print(f"Error processing folder {folder_name}:")
                # print type of error, line number, and error message
                print(traceback.format_exc())
            except Exception as e:
                print(f"Error processing folder {folder_name}:")
                print(traceback.format_exc())
    
    if processed_folders > 0:
        # Create a unified summary file
        output_path = os.path.join(main_folder_path, "summary_escape.dat")
        with open(output_path, 'w') as f:
            # Write header
            header = "Folder,EscapeRegime,EscapeTime_yrs"
            # Add other keys dynamically if needed, for now this is fixed
            f.write(header + "\n")
            
            for i, result in enumerate(summary_results):
                f.write(f"{names[i]} {result['escape_regime']} {result['escape_time_atmo_yrs']:.4e}\n")

        print("\nSummary of Escape Timescales:")
        escape_times = np.array([r['escape_time_atmo_yrs'] for r in summary_results if 'escape_time_atmo_yrs' in r and not np.isnan(r['escape_time_atmo_yrs'])])
        if len(escape_times) > 0:
            valid_names = np.array([names[i] for i, r in enumerate(summary_results) if 'escape_time_atmo_yrs' in r and not np.isnan(r['escape_time_atmo_yrs'])])
            print(f"Min: {valid_names[np.argmin(escape_times)]} - {np.min(escape_times):.2e} years")
            print(f"Max: {valid_names[np.argmax(escape_times)]} - {np.max(escape_times):.2e} years")
            print(f"Mean: {np.mean(escape_times):.2e} years")
        else:
            print("No valid escape times to summarize.")

    if processed_folders == 0:
        print("Warning: No folders with simulation data found in the provided path.")