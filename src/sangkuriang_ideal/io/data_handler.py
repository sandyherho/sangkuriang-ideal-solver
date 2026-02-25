# io/data_handler.py
"""NetCDF data handler for KdV simulation results."""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:
    """NetCDF output handler for KdV simulations."""
    
    @staticmethod
    def save_netcdf(filename: str, result: dict, metadata: dict,
                   output_dir: str = "outputs"):
        """
        Save KdV simulation results to NetCDF file.
        
        Structure:
            dimensions: x, t
            variables:
                x(x): Position [m]
                t(t): Time [s]
                u(t,x): Wave amplitude [m]
                mass(t): Conserved mass
                momentum(t): Conserved momentum
                energy(t): Conserved energy
            
            attributes:
                Physical parameters (μ, ε)
                Conservation errors
                Simulation metadata
        
        Args:
            filename: Output filename
            result: Results from solver.solve()
            metadata: Configuration dictionary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            
            # Dimensions
            nx = len(result['x'])
            nt = len(result['t'])
            nc.createDimension('x', nx)
            nc.createDimension('t', nt)
            
            # Coordinates
            nc_x = nc.createVariable('x', 'f4', ('x',), zlib=True, complevel=4)
            nc_x[:] = result['x']
            nc_x.units = "m"
            nc_x.long_name = "position"
            nc_x.axis = "X"
            
            nc_t = nc.createVariable('t', 'f4', ('t',), zlib=True, complevel=4)
            nc_t[:] = result['t']
            nc_t.units = "s"
            nc_t.long_name = "time"
            nc_t.axis = "T"
            
            # Wavefunction
            nc_u = nc.createVariable('u', 'f4', ('t', 'x'), zlib=True, complevel=5)
            nc_u[:] = result['u']
            nc_u.units = "m"
            nc_u.long_name = "wave_amplitude"
            nc_u.description = "KdV wave amplitude u(x,t)"
            
            # Conserved quantities
            nc_mass = nc.createVariable('mass', 'f4', ('t',), zlib=True, complevel=4)
            nc_mass[:] = result['mass']
            nc_mass.units = "m²"
            nc_mass.long_name = "conserved_mass"
            nc_mass.description = "M = ∫ u dx"
            
            nc_momentum = nc.createVariable('momentum', 'f4', ('t',), 
                                          zlib=True, complevel=4)
            nc_momentum[:] = result['momentum']
            nc_momentum.units = "m³"
            nc_momentum.long_name = "conserved_momentum"
            nc_momentum.description = "P = ∫ u² dx"
            
            nc_energy = nc.createVariable('energy', 'f4', ('t',), 
                                        zlib=True, complevel=4)
            nc_energy[:] = result['energy']
            nc_energy.units = "m⁴"
            nc_energy.long_name = "conserved_energy"
            nc_energy.description = "E = ∫ ((ε/2)·u³ - (3μ/2)·u_x²) dx"
            
            # Conservation errors
            nc.mass_error = float(result['mass_error'])
            nc.mass_error_description = "Max relative error in mass"
            
            nc.momentum_error = float(result['momentum_error'])
            nc.momentum_error_description = "Max relative error in momentum"
            
            nc.energy_error = float(result['energy_error'])
            nc.energy_error_description = "Max relative error in energy"
            
            # Physical parameters
            params = result['params']
            nc.mu = float(params['mu'])
            nc.mu_units = "m³/s"
            nc.mu_description = "Dispersion coefficient"
            
            nc.epsilon = float(params['eps'])
            nc.epsilon_units = "1/m"
            nc.epsilon_description = "Nonlinearity parameter"
            
            # Numerical parameters
            nc.nx = int(params['nx'])
            nc.dx = float(params['dx'])
            nc.dx_units = "m"
            
            nc.n_steps = int(params['n_steps'])
            nc.rtol = float(params['rtol'])
            nc.atol = float(params['atol'])
            
            nc.n_cores = int(params['n_cores'])
            nc.numba_enabled = int(params['numba_enabled'])
            
            # Soliton properties
            nc.soliton_velocity = float(params['soliton_velocity'])
            nc.soliton_velocity_units = "m/s"
            nc.soliton_velocity_description = "Theoretical velocity = ε·A/3"
            
            # Scenario info
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.initial_condition_type = metadata.get('ic_type', 'unknown')
            
            # Provenance
            nc.created = datetime.now().isoformat()
            nc.software = "sangkuriang-ideal-solver"
            nc.version = "0.0.7"
            nc.method = "spectral_fourier_dop853"
            nc.method_description = "Pseudo-spectral Fourier derivatives + DOP853 time integration"
            
            nc.Conventions = "CF-1.8"
            nc.title = f"KdV Soliton Simulation: {metadata.get('scenario_name', 'unknown')}"
            nc.institution = "Applied Geology Research Group ITB"
            nc.license = "MIT"
            nc.history = f"Created {datetime.now().isoformat()}"
