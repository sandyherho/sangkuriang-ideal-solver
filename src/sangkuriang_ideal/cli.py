#!/usr/bin/env python
"""
Command Line Interface for sangkuriang KdV Solver
four test cases with increasing complexity
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from .core.solver import KdVSolver
from .core.initial_conditions import SechProfile, MultiSoliton
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "sangkuriang: Idealized KdV Solver")
    print(" " * 22 + "Version 0.0.7")
    print("=" * 70)
    print("\n  Korteweg-de Vries Soliton Solver")
    print(" Pseudo-spectral Methods + Adaptive Time Stepping")
    print("\n  License: MIT ")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    if clean.startswith('case_'):
        parts = clean.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            case_num = parts[1]
            rest = '_'.join(parts[2:])
            clean = f"case{case_num}_{rest}"
    
    clean = clean.rstrip('_')
    return clean


def create_initial_condition(config: dict, x: np.ndarray) -> np.ndarray:
    """Create initial condition based on configuration."""
    ic_type = config.get('ic_type', 'sech')
    
    if ic_type == 'sech':
        ic = SechProfile(
            amplitude=config.get('amplitude', 4.0),
            width=config.get('width', 2.0),
            position=config.get('position', -10.0)
        )
        return ic(x)
    
    elif ic_type == 'multi_soliton':
        n_solitons = config.get('n_solitons', 2)
        amplitudes = []
        widths = []
        positions = []
        
        for i in range(1, n_solitons + 1):
            amplitudes.append(config.get(f'amplitude_{i}', 3.0))
            widths.append(config.get(f'width_{i}', 2.0))
            positions.append(config.get(f'position_{i}', -10.0 + i * 10.0))
        
        ic = MultiSoliton(amplitudes, widths, positions)
        return ic(x)
    
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")


def run_scenario(config: dict, output_dir: str = "outputs",
                verbose: bool = True, n_cores: int = None):
    """Run a complete KdV simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        # [1/5] Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing solver...")
            
            solver = KdVSolver(
                nx=config.get('nx', 512),
                x_min=config.get('x_min', -30.0),
                x_max=config.get('x_max', 30.0),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores
            )
        
        # [2/5] Create initial condition
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Creating initial condition...")
            
            u0 = create_initial_condition(config, solver.x)
            
            if verbose:
                ic_type = config.get('ic_type', 'sech')
                print(f"      Type: {ic_type}")
                if ic_type == 'multi_soliton':
                    n = config.get('n_solitons', 1)
                    print(f"      Number of solitons: {n}")
        
        # [3/5] Solve
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/5] Solving KdV equation...")
            
            result = solver.solve(
                u0=u0,
                mu=config.get('mu', 0.1),
                eps=config.get('eps', 0.2),
                t_final=config.get('t_final', 50.0),
                rtol=config.get('rtol', 1e-10),
                atol=config.get('atol', 1e-12),
                n_snapshots=config.get('n_frames', 200)
            )
            
            logger.log_results(result)
            
            if verbose:
                print(f"\n      Conservation Errors:")
                print(f"        Mass: {result['mass_error']:.2e}")
                print(f"        Momentum: {result['momentum_error']:.2e}")
                print(f"        Energy: {result['energy_error']:.2e}")
        
        # Get simulation time before continuing
        sim_time = timer.times.get('simulation', 0)
        
        # [4/5] Save NetCDF
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving NetCDF data...")
                
                filename = f"{clean_name}.nc"
                DataHandler.save_netcdf(filename, result, config, output_dir)
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        # [5/5] Create animation
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                if verbose:
                    print("\n[5/5] Creating animation...")
                
                filename = f"{clean_name}.gif"
                
                Animator.create_gif(
                    result,
                    filename,
                    output_dir,
                    scenario_name,
                    fps=config.get('fps', 30),
                    dpi=config.get('dpi', 150),
                    view_3d=config.get('view_3d', True),
                    colormap=config.get('colormap', 'plasma'),
                    line_width=config.get('line_width', 2.5),
                    alpha=config.get('alpha', 0.9)
                )
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        # Get timing breakdown
        anim_time = timer.times.get('animation', 0)
        total_time = timer.times.get('total', 0)
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Simulation time: {sim_time:.2f} s")
            print(f"  Animation time: {anim_time:.2f} s")
            print(f"  Total time: {total_time:.2f} s")
            
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 60}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 60}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Sangkuriang: Idealized KdV Soliton Solver',
        epilog='Example: sangkuriang case1 --cores 8'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (case1-4 with increasing complexity)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: all available)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    # Custom config
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    # All cases
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    
    # Single case
    elif args.case:
        case_map = {
            'case1': 'case1_single_soliton',
            'case2': 'case2_two_solitons',
            'case3': 'case3_collision',
            'case4': 'case4_three_solitons'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
