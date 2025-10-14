# `sangkuriang`: Idealized KdV solver
[![DOI](https://zenodo.org/badge/1075146518.svg)](https://doi.org/10.5281/zenodo.17350032)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/sangkuriang-ideal-solver.svg)](https://pypi.org/project/sangkuriang-ideal-solver/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

High-performance idealized Korteweg-de Vries (KdV) soliton solver with spectral methods and Numba acceleration.

## Physics

Solves the Korteweg-de Vries equation:

$$\frac{\partial u}{\partial t} + \varepsilon \cdot u \cdot \frac{\partial u}{\partial x} + \mu \cdot \frac{\partial^3 u}{\partial x^3} = 0$$

**Conservation Laws:** Mass, Momentum, Energy

## Features

- Spectral (Fourier) spatial derivatives
- DOP853 adaptive 8th-order Runge-Kutta
- Numba JIT compilation (10-100× speedup)
- Multi-core parallel processing
- 3D animated visualizations
- NetCDF4 output

## Installation

**From PyPI:**
```bash
pip install sangkuriang-ideal-solver
```

**From source:**
```bash
git clone https://github.com/sandyherho/sangkuriang-ideal-solver.git
cd sangkuriang-ideal-solver
pip install -e .
```

## Quick Start

**Command line:**
```bash
sangkuriang case1           # Single soliton
sangkuriang case3 --cores 8 # Collision with 8 cores
sangkuriang --all           # Run all cases
```

**Python API:**
```python
from sangkuriang_ideal import KdVSolver, SechProfile

solver = KdVSolver(nx=512, x_min=-30.0, x_max=30.0, n_cores=8)
u0 = SechProfile(amplitude=4.0, width=2.0, position=-10.0)(solver.x)
result = solver.solve(u0=u0, mu=0.1, eps=0.2, t_final=50.0)

print(f"Energy error: {result['energy_error']:.2e}")
```

## Test Cases

| Case | Description | Physics |
|------|-------------|---------|
| 1 | Single soliton | Baseline |
| 2 | Two equal solitons | Phase shift |
| 3 | Collision (different heights) | Overtaking |
| 4 | Three-soliton system | Complex multi-body |

## Citation

If you use this software in your research, please cite:

```bibtex
@article{herho202x_sangkuriang,
  title   = {Sangkuriang: An Idealized {K}orteweg-de {V}ries Soliton Solver with Spectral Methods},
  author  = {Herho, Sandy H. S. and Irawan, Dasapta E. and Suwarman, Rusmawan and Kaban, Siti N.},
  journal = {xxx},
  volume  = {xxx},
  pages   = {xxx--xxx},
  year    = {202x},
  doi     = {10.xxxx/xxxxx}
}
```

## Authors

- Sandy H.S. Herho
- Dasapta E. Irawan
- Rusmawan Suwarman  
- Siti N. Kaban

## License

WTFPL - Do What The Fuck You Want To Public License
