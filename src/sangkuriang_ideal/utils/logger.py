# utils/logger.py
"""Simulation logger for KdV solver."""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Enhanced logger for KdV simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs",
                 verbose: bool = True):
        """Initialize simulation logger."""
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging."""
        logger = logging.getLogger(f"kdv_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: dict):
        """Log all simulation parameters."""
        self.info("=" * 60)
        self.info(f"PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        """Log timing breakdown."""
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        # Separate simulation and post-processing times
        sim_time = timing.get('simulation', 0)
        anim_time = timing.get('animation', 0)
        total_time = timing.get('total', 0)
        
        self.info(f"  Simulation: {sim_time:.3f} s")
        self.info(f"  Animation: {anim_time:.3f} s")
        
        for key, value in sorted(timing.items()):
            if key not in ['simulation', 'animation', 'total']:
                self.info(f"  {key}: {value:.3f} s")
        
        self.info(f"  Total: {total_time:.3f} s")
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        """Log simulation results with conservation checks."""
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        params = results['params']
        
        self.info(f"  Physical parameters:")
        self.info(f"    μ = {params['mu']} m³/s")
        self.info(f"    ε = {params['eps']} 1/m")
        
        self.info(f"  Numerical:")
        self.info(f"    Grid: {params['nx']} points, dx = {params['dx']:.6f} m")
        self.info(f"    Steps: {params['n_steps']}")
        self.info(f"    Cores: {params['n_cores']}")
        self.info(f"    Numba: {'ENABLED' if params['numba_enabled'] else 'DISABLED'}")
        
        # Conservation laws
        mass_err = results['mass_error']
        momentum_err = results['momentum_error']
        energy_err = results['energy_error']
        
        self.info(f"  Conservation laws:")
        self.info(f"    Mass error: {mass_err:.2e}")
        self.info(f"    Momentum error: {momentum_err:.2e}")
        self.info(f"    Energy error: {energy_err:.2e}")
        
        # Quality assessment
        if energy_err < 0.01:
            quality = "EXCELLENT"
        elif energy_err < 0.05:
            quality = "GOOD"
        elif energy_err < 0.10:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
            self.warning(f"High energy error: {energy_err:.2e}")
        
        self.info(f"  Conservation quality: {quality}")
        
        # Soliton properties
        velocity = params['soliton_velocity']
        self.info(f"  Soliton velocity: {velocity:.6f} m/s")
        
        self.info(f"  Energy formula: E = ∫ ((ε/2)·u³ - (3μ/2)·u_x²) dx")
        
        self.info("=" * 60)
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 60)
        self.info("SIMULATION SUMMARY")
        self.info("=" * 60)
        
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"    {i}. {err}")
        else:
            self.info("  ERRORS: None")
        
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"    {i}. {warn}")
        else:
            self.info("  WARNINGS: None")
        
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("=" * 60)
