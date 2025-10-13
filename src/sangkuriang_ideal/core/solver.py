"""
sangkuriang: Idealized 1D Korteweg-de Vries Soliton Solver
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Optional
from tqdm import tqdm
import warnings
import os

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range
    def set_num_threads(n):
        pass

warnings.filterwarnings('ignore')


@njit(parallel=True, cache=True)
def compute_nonlinear_term(u: np.ndarray, u_x: np.ndarray, eps: float) -> np.ndarray:
    """Compute nonlinear term with Numba parallelization."""
    result = np.empty_like(u)
    for i in prange(len(u)):
        result[i] = -eps * u[i] * u_x[i]
    return result


@njit(parallel=True, cache=True)
def compute_dispersion_term(u_xxx: np.ndarray, mu: float) -> np.ndarray:
    """Compute dispersion term with Numba parallelization."""
    result = np.empty_like(u_xxx)
    for i in prange(len(u_xxx)):
        result[i] = -mu * u_xxx[i]
    return result


class KdVSolver:
    """
    High-performance idealized KdV equation solver.
    
    Solves: ∂u/∂t + ε·u·∂u/∂x + μ·∂³u/∂x³ = 0
    
    Key features:
    - Spectral (Fourier) spatial derivatives (exponential accuracy)
    - DOP853 adaptive time integration (8th-order Runge-Kutta)
    - Numba JIT acceleration with parallel processing
    - Fixed energy conservation (uses spectral derivatives)
    - SI units: x[m], t[s], u[m], velocity[m/s]
    """
    
    def __init__(self, nx: int = 512, x_min: float = -30.0, x_max: float = 30.0,
                 verbose: bool = True, logger: Optional[Any] = None,
                 n_cores: Optional[int] = None):
        """
        Initialize KdV solver.
        
        Args:
            nx: Number of spatial grid points
            x_min: Left boundary [m]
            x_max: Right boundary [m]
            verbose: Print progress messages
            logger: Optional logger instance
            n_cores: Number of CPU cores (None = all available)
        """
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.verbose = verbose
        self.logger = logger
        
        # Setup grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        
        # Setup wavenumbers for spectral derivatives
        self.k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        
        # Setup parallel processing
        if n_cores is None:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.n_cores)
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.6f} m")
            print(f"  Domain: [{x_min:.1f}, {x_max:.1f}] m")
            print(f"  CPU cores: {self.n_cores}")
            print(f"  Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    
    def spatial_derivative(self, u: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute spatial derivatives using spectral method.
        
        Args:
            u: Field values
            order: Derivative order (1 or 3)
        
        Returns:
            Derivative of u
        """
        u_hat = np.fft.fft(u)
        
        if order == 1:
            du_hat = 1j * self.k * u_hat
        elif order == 3:
            du_hat = (1j * self.k)**3 * u_hat
        else:
            raise ValueError(f"Unsupported derivative order: {order}")
        
        return np.real(np.fft.ifft(du_hat))
    
    def kdv_rhs(self, t: float, u: np.ndarray, mu: float, eps: float) -> np.ndarray:
        """
        Right-hand side of KdV equation.
        
        ∂u/∂t = -ε·u·∂u/∂x - μ·∂³u/∂x³
        """
        u_x = self.spatial_derivative(u, order=1)
        u_xxx = self.spatial_derivative(u, order=3)
        
        if NUMBA_AVAILABLE:
            nonlinear = compute_nonlinear_term(u, u_x, eps)
            dispersion = compute_dispersion_term(u_xxx, mu)
            dudt = nonlinear + dispersion
        else:
            dudt = -eps * u * u_x - mu * u_xxx
        
        return dudt
    
    def compute_invariants(self, u_hist: np.ndarray, mu: float) -> Dict[str, np.ndarray]:
        """
        Compute conserved quantities using spectral derivatives (FIXED).
        
        Conservation laws:
        - Mass: M = ∫ u dx
        - Momentum: P = ∫ u² dx
        - Energy: E = ∫ (u³ - μ·u_x²) dx
        
        Args:
            u_hist: Solution history [n_snapshots × nx]
            mu: Dispersion coefficient
        
        Returns:
            Dictionary with mass, momentum, energy arrays
        """
        n_snapshots = u_hist.shape[0]
        
        mass = np.zeros(n_snapshots)
        momentum = np.zeros(n_snapshots)
        energy = np.zeros(n_snapshots)
        
        for i in range(n_snapshots):
            u = u_hist[i]
            
            # Mass: ∫ u dx
            mass[i] = np.trapz(u, self.x)
            
            # Momentum: ∫ u² dx
            momentum[i] = np.trapz(u**2, self.x)
            
            # Energy: ∫ (u³ - μ·u_x²) dx
            # FIXED: Use spectral derivative for consistency
            u_x = self.spatial_derivative(u, order=1)
            energy[i] = np.trapz(u**3 - mu * u_x**2, self.x)
        
        return {
            'mass': mass,
            'momentum': momentum,
            'energy': energy
        }
    
    def solve(self, u0: np.ndarray, mu: float, eps: float,
              t_final: float = 50.0, rtol: float = 1e-10, atol: float = 1e-12,
              n_snapshots: int = 200) -> Dict[str, Any]:
        """
        Solve KdV equation with adaptive time stepping.
        
        Args:
            u0: Initial condition [m]
            mu: Dispersion coefficient [m³/s]
            eps: Nonlinearity parameter [1/m]
            t_final: Final time [s]
            rtol: Relative tolerance
            atol: Absolute tolerance
            n_snapshots: Number of output snapshots
        
        Returns:
            Dictionary with solution and diagnostics
        """
        if self.verbose:
            print(f"\n  Solving KdV equation...")
            print(f"    μ = {mu} m³/s, ε = {eps} 1/m")
            print(f"    t ∈ [0, {t_final}] s")
            print(f"    Method: DOP853 (8th-order RK)")
            print(f"    Tolerances: rtol={rtol}, atol={atol}")
        
        # Time points for output
        t_eval = np.linspace(0, t_final, n_snapshots)
        
        # Progress tracking
        if self.verbose:
            pbar = tqdm(
                total=t_final,
                desc="  Integrating",
                unit="s",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}]'
            )
            last_t = 0.0
            
            def kdv_rhs_progress(t, u):
                nonlocal last_t
                if t > last_t:
                    pbar.update(t - last_t)
                    last_t = t
                return self.kdv_rhs(t, u, mu, eps)
            
            rhs_func = kdv_rhs_progress
        else:
            rhs_func = lambda t, u: self.kdv_rhs(t, u, mu, eps)
        
        # Solve ODE
        solution = solve_ivp(
            rhs_func,
            (0, t_final),
            u0,
            method='DOP853',
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False
        )
        
        if self.verbose:
            pbar.close()
        
        if not solution.success:
            error_msg = f"Integration failed: {solution.message}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        t = solution.t
        u = solution.y.T  # Shape: [n_snapshots × nx]
        
        if self.verbose:
            print(f"  ✓ Solution computed ({solution.nfev} evaluations)")
        
        # Compute conserved quantities
        if self.verbose:
            print(f"  Computing conservation laws...")
        
        invariants = self.compute_invariants(u, mu)
        
        # Calculate relative errors
        mass = invariants['mass']
        momentum = invariants['momentum']
        energy = invariants['energy']
        
        mass_error = np.abs((mass - mass[0]) / mass[0]).max()
        momentum_error = np.abs((momentum - momentum[0]) / momentum[0]).max()
        energy_error = np.abs((energy - energy[0]) / energy[0]).max()
        
        if self.verbose:
            print(f"    Mass error: {mass_error:.2e}")
            print(f"    Momentum error: {momentum_error:.2e}")
            print(f"    Energy error: {energy_error:.2e}")
            
            if energy_error < 0.01:
                print(f"    ✓ Energy conservation: EXCELLENT")
            elif energy_error < 0.05:
                print(f"    ✓ Energy conservation: GOOD")
            else:
                print(f"    ⚠ Energy conservation: Check parameters")
        
        # Calculate soliton velocity for tracking
        # For sech² profile: v = ε·A/3
        amplitude = np.max(u[0])
        velocity = eps * amplitude / 3.0
        
        return {
            'x': self.x,
            't': t,
            'u': u,
            'mass': mass,
            'momentum': momentum,
            'energy': energy,
            'mass_error': mass_error,
            'momentum_error': momentum_error,
            'energy_error': energy_error,
            'params': {
                'mu': mu,
                'eps': eps,
                'nx': self.nx,
                'dx': self.dx,
                'n_steps': solution.nfev,
                'rtol': rtol,
                'atol': atol,
                'n_cores': self.n_cores,
                'numba_enabled': NUMBA_AVAILABLE,
                'soliton_velocity': velocity
            }
        }
