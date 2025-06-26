import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any, Union
from matplotlib.colors import LogNorm, SymLogNorm
from .data_loader import ChelioRun

def _get_profile_data(data: dict, chelio_run: ChelioRun, param_key: str, mol_type: str = 'mol'):
    """Helper function to extract a data profile based on a key."""
    
    mol = mol_type == 'mol'
    
    dust = mol_type == 'dust'
    supersat = mol_type == 'supersat'
    dust = dust or supersat
    
    atom = mol_type == 'atom'
    eps = mol_type == 'eps'
    atom = atom or eps
    
    if param_key.lower() == 'temperature':
        return data['temperature_K'], "Temperature [K]"
    elif param_key in chelio_run.mol_names and mol:
        idx = chelio_run.mol_names.index(param_key)
        return data['mols_vmr'][:, idx], f"Volume Mixing Ratio of {param_key}"
    elif param_key in chelio_run.dust_names and dust:
        idx = chelio_run.dust_names.index(param_key)
        # Check which dust data to plot, default to VMR
        if supersat:
             return data['supersats'][:, idx], f"Supersaturation Ratio of {param_key}"
        return data['dusts_vmr'][:, idx], f"Mixing Ratio of {param_key}"
    elif param_key in chelio_run.atom_names and atom:
        idx = chelio_run.atom_names.index(param_key)
        if eps:
            idx = idx - 1 # remove electron (first entry of atom_names)
            return data['eps_atoms_mr'][:, idx], f"Elemental Abundance of {param_key}"
        return data['atoms_vmr'][:, idx], f"Mixing Ratio of {param_key}"
    elif param_key in vars(chelio_run).keys():
        return data[param_key], f"{param_key}"
    
    raise ValueError(f"param_key '{param_key}' is not a valid plotting key.")

def _setup_profile_axes(ax, y_axis, x_label, log_x, log_y):
    """Configures the axes for a profile plot. This should be called only once per plot."""
    if y_axis == 'pressure':
        ax.set_ylabel("Pressure [bar]")
        # This check prevents flipping the axis back and forth.
        if not ax.yaxis_inverted():
                ax.invert_yaxis()
    elif y_axis == 'altitude':
        ax.set_ylabel("Altitude [cm]")
    else:
        raise ValueError("y_axis must be 'pressure' or 'altitude'")
    
    ax.set_xlabel(x_label)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')


def plot_profile(
    chelio_run: ChelioRun, 
    param_key: Union[str, List[str]], 
    y_axis: str = 'pressure', 
    log_y: bool = True, 
    log_x: bool = False,
    ax=None, 
    iteration_index: int = -1, 
    mol_type: str = 'mol',
    **kwargs
):
    """
    Plots single or multiple profiles (e.g., temperature, VMR) vs. pressure or altitude for a single run.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not chelio_run.is_converted:
        chelio_run.convert_to_vmr()

    data = chelio_run.get_iteration_data(iteration_index)
    if "error" in data:
        print(f"Cannot plot: {data['error']}")
        return fig, ax
    
    # Determine Y-axis data
    if y_axis == 'pressure':
        y_data = data['pressure_bar']
    elif y_axis == 'altitude':
        y_data = data['altitudes_cm']
    else:
        raise ValueError("y_axis must be 'pressure' or 'altitude'")

    param_keys = [param_key] if isinstance(param_key, str) else param_key
    
    # Setup axes
    if len(param_keys) == 1:
        _, x_label = _get_profile_data(data, chelio_run, param_keys[0], mol_type)
    else:
        x_label = "Value"
    _setup_profile_axes(ax, y_axis, x_label, log_x, log_y)

    # Pop label to avoid applying it to every line in the loop
    label_kwarg = kwargs.pop('label', None)

    for key in param_keys:
        x_data, _ = _get_profile_data(data, chelio_run, key, mol_type)
        
        # Determine the label for this line
        label = label_kwarg if len(param_keys) == 1 else key
        if label_kwarg and len(param_keys) > 1:
             label = f"{label_kwarg} - {key}"
        
        if label in chelio_run.mol_names or label in chelio_run.dust_names or label in chelio_run.supersats:
            for number in np.arange(10, dtype=int):
                label = label.replace(f"{number}", "$_{" + f"{number}" + "}$")

        ax.plot(x_data, y_data, label=label, **kwargs)

    # Add legend if multiple lines were plotted
    if len(param_keys) > 1 and label_kwarg is None:
        ax.legend()
        
    return fig, ax

def plot_profile_comparison(
    run_list: List[ChelioRun],
    param_key: str,
    y_axis: str = 'pressure',
    log_y: bool = True,
    log_x: bool = False,
    ax=None,
    labels: List[str] = None,
    cmap_name='viridis',
    mol_type: str = 'mol',
    **kwargs
):
    """
    Plots a comparison of a single profile across multiple ChelioRun instances.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Setup axes once, based on the properties of the first run.
    if run_list:
        first_run = run_list[0]
        if not first_run.is_converted:
            first_run.convert_to_vmr()
        data = first_run.get_iteration_data()
        if "error" not in data:
            _, x_label = _get_profile_data(data, first_run, param_key, mol_type)
            _setup_profile_axes(ax, y_axis, x_label, log_x, log_y)

    cmap = plt.get_cmap(cmap_name)
    
    for i, run in enumerate(run_list):
        if not run.is_converted:
            run.convert_to_vmr()
        data = run.get_iteration_data()
        if "error" in data:
            continue
        
        if y_axis == 'pressure':
            y_data = data['pressure_bar']
        else:
            y_data = data['altitudes_cm']
        
        x_data, _ = _get_profile_data(data, run, param_key, mol_type)
        
        label = labels[i] if labels is not None and i < len(labels) else f"Run {i}"
        color = cmap(i / max(1, len(run_list) - 1))
        
        ax.plot(x_data, y_data, label=label, color=color, **kwargs)
    
    ax.legend(title=kwargs.get('legend_title'))
    return fig, ax


def plot_all_iteration_profiles(
    chelio_run: ChelioRun, 
    param_key: str, 
    y_axis: str = 'pressure', 
    log_y: bool = True, 
    log_x: bool = False,
    ax=None, 
    cmap_name='viridis', 
    mol_type: str = 'mol',
    **kwargs
):
    """
    Plots the evolution of a profile over all coupling iterations for a single run.
    """
    if chelio_run.load_mode != 'all':
        print("Warning: ChelioRun was not loaded with load_mode='all'. Plot may be incomplete.")
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Setup axes once, based on the last iteration.
    if chelio_run.num_iterations_read > 0:
        if not chelio_run.is_converted:
            chelio_run.convert_to_vmr()
        data = chelio_run.get_iteration_data(-1)
        if "error" not in data:
            _, x_label = _get_profile_data(data, chelio_run, param_key, mol_type)
            _setup_profile_axes(ax, y_axis, x_label, log_x, log_y)

    cmap = plt.get_cmap(cmap_name)
    num_iters = chelio_run.num_iterations_read
    
    for i in range(num_iters):
        data = chelio_run.get_iteration_data(i)
        if "error" in data:
            continue

        if y_axis == 'pressure':
            y_data = data['pressure_bar']
        else:
            y_data = data['altitudes_cm']

        x_data, _ = _get_profile_data(data, chelio_run, param_key, mol_type)
        
        color = cmap(i / max(1, num_iters - 1))
        label = f"{i}"
        
        ax.plot(x_data, y_data, color=color, label=label, **kwargs)
    
    ax.legend()
    return fig, ax

def _get_scalar_data(run: ChelioRun, y_param_key: str, layer_idx: int):
    """Helper to extract a single scalar data point from a run."""
    if not run.final_convergence_status:
        return np.nan
        
    if not run.is_converted:
        run.convert_to_vmr()
            
    data = run.get_iteration_data()
        
    if y_param_key in run.mol_names:
        mol_idx = run.mol_names.index(y_param_key)
        return data['mols_vmr'][layer_idx, mol_idx]
    elif y_param_key.lower() == 't_surf':
        return data['temperature_K'][0]
    elif y_param_key.lower() == 'escape_time':
        return run.escape_time_yrs
    # Add more extraction options here as needed
    
    raise ValueError(f"y_param_key '{y_param_key}' is not a valid scalar key.")


def plot_comparison(
    run_list: List[ChelioRun], 
    x_param_values: List[Any],
    y_param_key: str, 
    x_param_name: str,
    layer_idx: int = 0,
    log_y: bool = True,
    log_x: bool = False,
    ax=None,
    **kwargs
):
    """
    Plots an output parameter against a varying input parameter from a sweep.
    e.g., surface abundance of H2O vs. C/O ratio.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    y_values = [_get_scalar_data(run, y_param_key, layer_idx) for run in run_list]
        
    ax.plot(x_param_values, y_values, '.-', **kwargs)
    
    ax.set_xlabel(x_param_name)
    y_label = y_param_key
    if layer_idx == 0:
        y_label += " (surface)"
    ax.set_ylabel(y_label)
    
    if log_x: ax.set_xscale('log')
    if log_y: ax.set_yscale('log')
    
    return fig, ax

def plot_2d_matrix(
    data_matrix: np.ndarray, 
    param1_values: List[Any], 
    param2_values: List[Any],
    param1_name: str,
    param2_name: str,
    z_label: str,
    ax=None, 
    log_z: bool = False,
    symlog_z: bool = False,
    display_text: bool = False,
    text_fmt: str = ".2f",
    **kwargs
):
    """
    Generates a heatmap/imshow plot of a scalar result over a 2D parameter grid.
    Supports logarithmic color scaling and displaying values in cells.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    norm = None
    if log_z:
        norm = LogNorm(vmin=kwargs.pop('vmin', None), vmax=kwargs.pop('vmax', None))
    elif symlog_z:
        norm = SymLogNorm(linthresh=kwargs.pop('linthresh', 1e-5), vmin=kwargs.pop('vmin', None), vmax=kwargs.pop('vmax', None))
        
    im = ax.imshow(data_matrix.T, origin='lower', aspect='auto', norm=norm, **kwargs)
    
    fig.colorbar(im, ax=ax, label=z_label)
    
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_xticklabels([f"{v:.2g}" for v in param1_values])
    ax.set_xlabel(param1_name)
    
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_yticklabels([f"{v:.2g}" for v in param2_values])
    ax.set_ylabel(param2_name)

    if display_text:
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                val = data_matrix[i, j]
                if not np.isnan(val):
                    ax.text(i, j, format(val, text_fmt), ha="center", va="center", color="black")
    
    return fig, ax

def plot_rcb_height(
    chelio_run: ChelioRun,
    ax=None,
    iteration_index: int = -1,
    **kwargs
):
    """
    Finds and plots the radiative-convective boundary (RCB).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    data = chelio_run.get_iteration_data(iteration_index)
    if "error" in data:
        print(f"Cannot plot: {data['error']}")
        return fig, ax

    convective_flags = data['convective_flag']
    pressures = data['pressure_bar']
    
    # Find first radiative layer above a convective one
    rcb_index = np.where(np.diff(convective_flags.astype(int)) == -1)[0]
    
    if rcb_index.size > 0:
        i = rcb_index[0]
        # Interpolate pressure to find RCB
        rcb_pressure = 10**((np.log10(pressures[i]) + np.log10(pressures[i+1])) / 2)
        xmin, xmax = ax.get_xlim()
        ax.hlines(rcb_pressure, xmin, xmax, **kwargs)
    else:
        print("RCB not found in the given data.")
    
    return fig, ax 