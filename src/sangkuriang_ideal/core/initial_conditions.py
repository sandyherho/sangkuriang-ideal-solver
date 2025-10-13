"""
Initial condition profiles for KdV soliton solver.

Provides exact and approximate soliton solutions:
- SechProfile: Exact single-soliton solution (sech²)
- TanhProfile: Shock-wave-like profile
- GaussianProfile: Approximate localized wave
- MultiSoliton: Superposition of multiple solitons
"""

import numpy as np
from typing import List


class SechProfile:
    """
    Exact single-soliton solution: u(x,0) = A/cosh²((x-x₀)/w)
    
    For KdV equation with this profile:
    - Soliton velocity: v = ε·A/3
    - Width parameter: w (smaller = narrower)
    - Amplitude: A (higher = faster)
    """
    
    def __init__(self, amplitude: float = 4.0, width: float = 2.0, 
                 position: float = -10.0):
        """
        Args:
            amplitude: Soliton amplitude A [m]
            width: Soliton width w [m]
            position: Initial position x₀ [m]
        """
        self.amplitude = amplitude
        self.width = width
        self.position = position
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate sech² profile."""
        xi = (x - self.position) / self.width
        return self.amplitude / np.cosh(xi)**2
    
    def velocity(self, eps: float) -> float:
        """Calculate theoretical soliton velocity."""
        return eps * self.amplitude / 3.0


class TanhProfile:
    """
    Tanh profile: u(x,0) = (A/2)·[1 - tanh((x-x₀)/w)]
    
    Creates a shock-wave-like transition.
    """
    
    def __init__(self, amplitude: float = 4.0, width: float = 2.0,
                 position: float = -10.0):
        """
        Args:
            amplitude: Maximum height [m]
            width: Transition width [m]
            position: Center position [m]
        """
        self.amplitude = amplitude
        self.width = width
        self.position = position
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate tanh profile."""
        xi = 2.0 * (x - self.position) / self.width
        return 0.5 * self.amplitude * (1.0 - np.tanh(xi))


class GaussianProfile:
    """
    Gaussian profile: u(x,0) = A·exp(-(x-x₀)²/w²)
    
    Not an exact solution, will disperse into solitons + radiation.
    """
    
    def __init__(self, amplitude: float = 4.0, width: float = 2.0,
                 position: float = -10.0):
        """
        Args:
            amplitude: Peak amplitude [m]
            width: Gaussian width σ [m]
            position: Peak position [m]
        """
        self.amplitude = amplitude
        self.width = width
        self.position = position
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian profile."""
        return self.amplitude * np.exp(-((x - self.position)**2) / self.width**2)


class MultiSoliton:
    """
    Multiple soliton superposition.
    
    Creates initial condition with several solitons that will interact.
    Each soliton is a sech² profile with its own amplitude, width, position.
    """
    
    def __init__(self, amplitudes: List[float], widths: List[float],
                 positions: List[float]):
        """
        Args:
            amplitudes: List of soliton amplitudes [m]
            widths: List of soliton widths [m]
            positions: List of initial positions [m]
        """
        if not (len(amplitudes) == len(widths) == len(positions)):
            raise ValueError("All lists must have same length")
        
        self.n_solitons = len(amplitudes)
        self.solitons = [
            SechProfile(a, w, p) 
            for a, w, p in zip(amplitudes, widths, positions)
        ]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate multi-soliton profile (linear superposition)."""
        u = np.zeros_like(x)
        for soliton in self.solitons:
            u += soliton(x)
        return u
    
    def velocities(self, eps: float) -> List[float]:
        """Calculate velocities of all solitons."""
        return [s.velocity(eps) for s in self.solitons]
