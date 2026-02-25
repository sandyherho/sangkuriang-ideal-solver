"""Sangkuriang: Idealized KdV Soliton Solver"""

__version__ = "0.0.8"
__author__ = "Dasapta E. Irawan, Sandy H. S. Herho, Astyka Pamumpuni, Rendy D. Kartiko, Faruq Khadami, and Iwan P. Anwar"
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
