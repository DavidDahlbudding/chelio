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

def radius_profile(P, r0, P0, T, mu, g):
    return r0 + (kB * T) / (mu * m_H * g) * np.log(P0 / P)

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

    # Load data using np.loadtxt, skip header and dimension lines (4 rows)
    try:
        static_data = np.loadtxt(static_data_path, skiprows=3) # altitude (cm) in column 0, pressure (dyn/cm^2) in column 2, Temp (K) in column 2, nHtot in column 1
        vertical_mix_mu = np.loadtxt(vertical_mix_path, skiprows=1) # mu in column 3, skip 2 header rows
        tp_data_alt = np.loadtxt(tp_data_path, skiprows=2, usecols=3) # altitude (cm) in column 3
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data file: {e}")
    except ValueError as e: # Catch errors during data loading (e.g., wrong skiprows)
        raise ValueError(f"Error loading data from file: {e}. Check file format and skiprows settings.")

    altitudes = tp_data_alt[:] # cm
    pressures_dyn_cm2 = static_data[:, 2] # dyn/cm^2 (column index 2)
    temperatures = static_data[:, 0] # K (column index 2)
    n_tots = vertical_mix_mu[:, 2] # nHtot values
    mu = vertical_mix_mu[:, 3] # mu values

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

    # 2. Exobase Calculation (interpolation and masked pressure)
    mfp = get_mfp(pressures_dyn_cm2, temperatures, kin_dia_H2) # mean free path
    scale_height = get_scale_height(radius, temperatures, m_P, mu) # scale height
    r_c_values = r_exobase(pressures_dyn_cm2, m_P, mu, kin_dia_H2) # exobase radius

    while mfp[-1] < scale_height[-1]:
        # approximate P at exobase
        approx_P = r_c_values[-1]**2 * pressures_dyn_cm2[-1] / (radius[-1]**2)
        approx_P = 10**np.floor(np.log10(approx_P))
        print(f"Extending profile to P = {approx_P} dyn/cm^2")
        # extend profiles to new P
        radius, pressures_dyn_cm2, temperatures, mu, n_tots = extend_profile(radius, pressures_dyn_cm2, temperatures, approx_P, mu, m_P, n_tots)
        altitudes = radius - r_P
        # recalculate mfp, scale height, and exobase radius
        mfp = get_mfp(pressures_dyn_cm2, temperatures, kin_dia_H2)
        scale_height = get_scale_height(radius, temperatures, m_P, mu)
        r_c_values = r_exobase(pressures_dyn_cm2, m_P, mu, kin_dia_H2)

    # Interpolate functions
    f_rc = interp1d(radius, r_c_values, bounds_error=False, fill_value="extrapolate") # r_c as function of altitude
    f_P = interp1d(radius, np.log10(pressures_dyn_cm2), bounds_error=False, fill_value="extrapolate") # pressure in dyn/cm^2
    f_T = interp1d(radius, temperatures, bounds_error=False, fill_value="extrapolate")
    f_n = interp1d(radius, np.log10(n_tots), bounds_error=False, fill_value="extrapolate")
    f_mu = interp1d(radius, mu, bounds_error=False, fill_value="extrapolate")

    # get exobase radius
    i_exo = np.sum(mfp < scale_height)
    # interpolate exobase radius where mfp == scale height
    dsh = scale_height[i_exo] - scale_height[i_exo-1]
    dmfp = mfp[i_exo] - mfp[i_exo-1]
    dr = radius[i_exo] - radius[i_exo-1]
    r_exo = radius[i_exo-1] + (scale_height[i_exo-1] - mfp[i_exo-1]) * dr / (dmfp - dsh)

    #print(f"Exobase altitude: {r_exo:.2e} cm")
    #print(f"Exobase radius: {f_rc(r_exo):.2e} cm")

    r_c = f_rc(r_exo)
    P_c = 10**f_P(r_exo)
    T_c = f_T(r_exo)
    n_c = 10**f_n(r_exo)
    mu_c = f_mu(r_exo)

    #print(r_c, P_c, T_c, n_c, mu_c)

    # 3. Thermal Escape Condition
    T_max_exo = maxTexo(m_P, r_P, mu_c)
    #print(f"Max T_exo: {T_max_exo:.2f} K")
    thermal_escape_condition = T_c > T_max_exo

    # 4. Jeans Escape Rate and Timescale
    lambda_c, phi_jeans = escape_rate(r_c, P_c, T_c, m_P, mu_c) # Jeans escape rate in [mol cm^-2 s^-1]
    phi_jeans_tot = phi_jeans * 4 * np.pi * r_c**2 # total Jeans escape rate in [mol s^-1] (molecule per second)
    escape_time = 1 / phi_jeans_tot * avogadro / mu_c # escape timescale in s/g
    escape_time_yrs = escape_time / (60*60*24*365.25) # escape timescale in years/g

    grav = G * m_P / r_P**2 # cm/s^2
    M_atmo = pressures_dyn_cm2[0] * 4 * np.pi * r_c**2 / grav # g
    escape_time_atmo_yrs = escape_time_yrs * M_atmo # years
    print(escape_time_atmo_yrs.shape)
    print('Rough time until escape of entire atmosphere:')
    print(f'{escape_time_atmo_yrs:.2e} years')

    # 5. Save Results to escape.dat
    output_path = os.path.join(folder_path, folder_name, "escape.dat")
    with open(output_path, 'w') as f:
        f.write(f"Planet Name: {planet_name}\n")
        f.write(f"Exobase Altitude [cm]: {r_exo:.4e}\n")
        f.write(f"Exobase Radius (r_c) [cm]: {r_c:.4e}\n")
        f.write(f"Exobase Pressure (P_c) [dyn/cm^2]: {P_c:.4e}\n") # Saved in dyn/cm^2
        f.write(f"Exobase Temperature (T_c) [K]: {T_c:.2f}\n")
        f.write(f"Exobase Number Density (n_c) [cm^-3]: {n_c:.4e}\n")
        f.write(f"Thermal Escape Condition Met: {thermal_escape_condition}\n")
        f.write(f"Jeans Escape Parameter (lambda_c): {lambda_c:.4e}\n")
        f.write(f"Jeans Escape Rate [cm^-2 s^-1]: {phi_jeans:.4e}\n")
        f.write(f"Escape Timescale of entire Atmosphere [years]: {escape_time_atmo_yrs:.2e}\n")

    return { # Return a dictionary for potential further use
        "r_c": r_c,
        "P_c": P_c,
        "T_c": T_c,
        "n_c": n_c,
        "thermal_escape_condition": thermal_escape_condition,
        "lambda_c": lambda_c,
        "phi_jeans": phi_jeans,
        "escape_time_atmo_yrs": escape_time_atmo_yrs
    }


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
    summary_lambda_c = []
    summary_escape_time = []

    for folder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(subfolder_path): # Check if it's a directory
            try:
                print(f"Processing folder: {folder_name}")
                escape_results = calculate_escape_parameters(main_folder_path, folder_name)
                names.append(folder_name)
                summary_lambda_c.append(escape_results["lambda_c"])
                summary_escape_time.append(escape_results["escape_time_atmo_yrs"])
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
        summary_lambda_c = np.array(summary_lambda_c)
        print(f"Summary of Jeans Escape Parameters:")
        print(f"Min: {names[np.argmin(summary_lambda_c)]} - {np.min(summary_lambda_c):.2e}")
        print(f"Max: {names[np.argmax(summary_lambda_c)]} - {np.max(summary_lambda_c):.2e}")
        print(f"Mean: {np.mean(summary_lambda_c):.2e}")
        summary_escape_time = np.array(summary_escape_time)
        print(f"Summary of Escape Timescales:")
        print(f"Min: {names[np.argmin(summary_escape_time)]} - {np.min(summary_escape_time):.2e} years")
        print(f"Max: {names[np.argmax(summary_escape_time)]} - {np.max(summary_escape_time):.2e} years")
        print(f"Mean: {np.mean(summary_escape_time):.2e} years")

        output_path = os.path.join(main_folder_path, "summary_escape.dat")
        with open(output_path, 'w') as f:
            for i in range(len(names)):
                f.write(f"{names[i]} {summary_lambda_c[i]:.4e} {summary_escape_time[i]:.4e}\n")

    if processed_folders == 0:
        print("Warning: No folders with simulation data found in the provided path.")