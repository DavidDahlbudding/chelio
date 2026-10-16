#!/usr/bin/env python3
"""
Run a grid of Chelio simulations defined in a grid file (see grids/README.md), one after another.

    python run_grid.py grids/n2dom.yaml --dry-run
    python run_grid.py grids/cia_comparison.yaml --only cia_pair=N2-CH4,CO2-CO2
    python run_grid.py --list-cia-sources
    python run_grid.py --restore-cia-defaults

Runs share input files (see CLAUDE.md), so there is deliberately no parallel mode.
"""
import argparse
import contextlib
import datetime
import os
import shlex
import shutil
import sys
from itertools import groupby

from tqdm import tqdm

from chelio_sim import grid


def list_cia_sources():
    print("Available CIA sources by pair (* = default):")
    for pair, sources in groupby(grid.discover_cia_sources(), key=grid.cia_pair):
        print(f"\n{pair} [default: {grid.DEFAULT_CIA_SOURCES.get(pair, 'N/A')}]:")
        for source in sources:
            marker = " *" if source == grid.DEFAULT_CIA_SOURCES.get(pair) else ""
            print(f"  - {source}{marker}")


def log_failure(log_path, task, cmd, result):
    with open(log_path, "a") as f:
        f.write(f"--- FAILED: {task.name} ({datetime.datetime.now():%Y-%m-%d %H:%M:%S}, exit code {result.returncode}) ---\n")
        f.write(f"Command: {shlex.join(cmd)}\n")
        if result.stdout is not None:
            f.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run a grid of Chelio simulations defined in a grid file (grids/*.yaml).")
    parser.add_argument("grid_file", nargs="?", help="Grid file, e.g. grids/n2dom.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the runs and their commands.")
    parser.add_argument(
        "--only", action="append", metavar="KEY=VALUE[,VALUE...]",
        help="Only run tasks whose parameter KEY has one of the values (repeatable, e.g. --only cia_pair=N2-CH4 --only internal_temp=100).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show the output of each run instead of capturing it.")
    parser.add_argument("--list-cia-sources", action="store_true", help="List the CIA sources in hitran_cia/ and exit.")
    parser.add_argument("--restore-cia-defaults", action="store_true", help="Copy the default CIA sources into r50_kdistr and exit.")
    args = parser.parse_args()

    if args.list_cia_sources:
        list_cia_sources()
        return
    if args.restore_cia_defaults:
        grid.restore_default_cia()
        return
    if not args.grid_file:
        parser.error("a grid file is required")

    try:
        spec = grid.load_grid(args.grid_file)
        tasks = grid.order_tasks(spec, grid.filter_tasks(grid.expand_tasks(spec), args.only))
        if spec.get("warm_start") and tasks:
            grid.check_warm_start(spec, tasks[0])
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Grid file: {args.grid_file}")
    print(f"Runner: {grid.RUNNER_SCRIPTS[spec['runner']]}, output directory: {spec['out_dir']}")
    print(f"Labels (not passed to the runner): {', '.join(grid.label_keys(spec, tasks)) or '-'}")
    if spec.get("warm_start"):
        print(f"Warm start from the next-higher {spec['warm_start']} run in {spec['out_dir']}")
    print(f"Total number of simulations: {len(tasks)}\n")

    if args.dry_run:
        for task in tasks:
            print(task.name)
            print(f"    {shlex.join(grid.build_command(spec, task)[1:])}")
        return
    if not tasks:
        return

    out_dir = grid.CHELIO_ROOT / spec["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.grid_file, out_dir / f"grid_{os.path.basename(args.grid_file)}")
    failure_log = out_dir / "grid_failures.log"

    uses_cia = any("cia" in task.params for task in tasks)
    if uses_cia:
        print("Restoring default CIA sources before starting...")
        grid.restore_default_cia()

    status_names = {"done": [], "skipped": [], "failed": []}
    with tqdm(total=len(tasks), desc="Simulations") as bar:
        for source, group in groupby(tasks, key=lambda task: task.params.get("cia")):
            with grid.swapped_cia(source) if source else contextlib.nullcontext():
                for task in group:
                    init_pt_file = grid.find_warm_start(spec, task) if spec.get("warm_start") else None
                    if spec.get("warm_start"):
                        bar.write(f"Warm start for {task.name}: {init_pt_file or 'none found'}")
                    cmd = grid.build_command(spec, task, init_pt_file)
                    status, result = grid.run_task(cmd, verbose=args.verbose)
                    status_names[status].append(task.name)
                    if status == "failed":
                        log_failure(failure_log, task, cmd, result)
                        bar.write(f"FAILED (exit code {result.returncode}): {task.name}, see {failure_log} and the run's run.log")
                    else:
                        bar.write(f"{status.upper()}: {task.name}")
                    bar.update()

    print("\n--- Grid complete ---")
    print(f"Completed: {len(status_names['done'])}, skipped (already existed): {len(status_names['skipped'])}, failed: {len(status_names['failed'])}")
    if status_names["failed"]:
        print("\nFailed simulation names:")
        for name in status_names["failed"]:
            print(f"- {name}")


if __name__ == "__main__":
    main()
