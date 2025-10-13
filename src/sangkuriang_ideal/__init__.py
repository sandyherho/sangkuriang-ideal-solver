"""Sangkuriang: Idealized 1D KdV Soliton Solver"""

__version__ = "0.0.1"
__author__ = "Sandy H.S. Herho, Rusmawan Suwarman, Dasapta E. Irawan"
__license__ = "WTFPL"

from .core.solver import KdVSolver
from .core.initial_conditions import SechProfile, TanhProfile, GaussianProfile, MultiSoliton
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "KdVSolver",
    "SechProfile",
    "TanhProfile", 
    "GaussianProfile",
    "MultiSoliton",
    "ConfigManager",
    "DataHandler"
]
