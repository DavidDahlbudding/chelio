import subprocess
import logging
import sys

log = logging.getLogger(__name__)

def run_command(command, cwd, input=None):
    """
    Runs an external command, streaming its output live to the console and log
    while also capturing it for error reporting.
    """
    log.debug(f"Running command: {' '.join(command)} in {cwd}")
    try:
        # Popen starts the process without blocking and allows streaming
        with subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Use line-buffering
        ) as process:

            if input is not None:
                stdout_output, stderr_output = process.communicate(input=input)
                if stdout_output:
                    for line in stdout_output.strip().split('\n'):
                        if line:
                            log.info(line)
            else:
                # Read stdout line by line in real-time
                last_line = ""
                for line in process.stdout:
                    line = line.strip()
                    if line or last_line:
                        # print(line)      # Print live to the console
                        log.info(line)   # Also write to the log file
                    last_line = line
                
            # Wait for the process to finish to get the return code
            process.wait()

            # Check for errors after the process has finished
            if process.returncode != 0:
                # Read any error output
                error_output = process.stderr.read()
                log.error(f"Command '{' '.join(command)}' failed with exit code {process.returncode}.")
                if error_output:
                    log.error(f"STDERR:\n{error_output.strip()}")
                # Raise an exception to stop the main script
                raise subprocess.CalledProcessError(process.returncode, command, stderr=error_output)

        log.info(f"Command '{command[0]}' completed successfully.")

    except FileNotFoundError:
        log.error(f"Error: The command '{command[0]}' was not found. Is the path correct?")
        sys.exit(1)

def run_helios(helios_path, params):
    """Constructs and runs the HELIOS command."""
    log.info("--- Preparing to run HELIOS ---")
    command = [
        "python3", "-u", "helios.py",
        "-name", params['name'],
        "-output_directory", params['output_directory'],
        "-toa_pressure", str(params['toa_pressure']),
        "-boa_pressure", str(params['boa_pressure']),
        "-internal_temperature", str(params['internal_temperature']),
        "-surface_albedo", str(params['surface_albedo']),
        "-opacity_mixing", "on-the-fly",
        "-path_to_species_file", params['path_to_species_file'],
        "-coupling_mode", params['coupling_mode'],
        "-coupling_iteration_step", str(params['coupling_iteration_step']),
        "-coupling_speed_up", params['coupling_speed_up'],
        "-write_tp_profile_during_run", str(params['write_tp_profile_during_run']),
        "-maximum_number_of_iterations", str(params['maximum_number_of_iterations']),
        "-radiative_equilibrium_criterion", str(params['radiative_equilibrium_criterion'])
    ]
    if 'path_to_temperature_file' in params:
        command.extend(["-path_to_temperature_file", params['path_to_temperature_file']])
    if 'file_with_vertical_mixing_ratios' in params:
        command.extend(["-file_with_vertical_mixing_ratios", params['file_with_vertical_mixing_ratios']])
    if 'kappa_value' in params:
        command.extend(["-kappa_value", str(params['kappa_value'])])
    if 'kappa_file_path' in params:
        command.extend(["-kappa_file_path", params['kappa_file_path']])
    if 'started_convection' in params:
         command.extend(["-started_convection", str(params['started_convection'])])

    run_command(command, cwd=helios_path)

def run_ggchem(ggchem_path, params_file="input/param_helios.in", input=None):
    """Runs the GGchem executable."""
    log.info("--- Preparing to run GGchem ---")
    command = ["./ggchem", params_file]
    run_command(command, cwd=ggchem_path, input=input)
