# `sangkuriang`: an idealized KdV solver

[![DOI](https://zenodo.org/badge/1075146518.svg)](https://doi.org/10.5281/zenodo.17350032)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/sangkuriang-ideal-solver.svg)](https://pypi.org/project/sangkuriang-ideal-solver/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

## Overview

`sangkuriang` is a high-performance Python solver for the idealized Korteweg-de Vries (KdV) equation, a fundamental nonlinear partial differential equation that describes the propagation of weakly nonlinear and weakly dispersive waves. The KdV equation is renowned for supporting solitons—localized wave packets that maintain their shape while traveling at constant velocity and exhibit remarkable stability during collisions.

This solver implements pseudo-spectral methods combined with adaptive high-order time integration to achieve spectral accuracy in space and exceptional temporal precision. The implementation leverages Numba's JIT compilation for significant performance gains (10-100× speedup) and supports multi-core parallel processing for large-scale simulations. The package includes comprehensive visualization tools for creating publication-quality 3D animations and NetCDF4 output for interoperability with scientific data analysis workflows.

**Key Applications:**
- Nonlinear wave dynamics research
- Soliton interaction studies
- Numerical methods benchmarking
- Educational demonstrations of wave phenomena
- Validation of theoretical predictions in nonlinear physics

## Physics

The solver addresses the canonical form of the Korteweg-de Vries equation:

$$\frac{\partial u}{\partial t} + \varepsilon \cdot u \cdot \frac{\partial u}{\partial x} + \mu \cdot \frac{\partial^3 u}{\partial x^3} = 0$$

where:
- $u(x,t)$ is the wave amplitude [m]
- $\varepsilon$ is the nonlinearity parameter [m⁻¹]
- $\mu$ is the dispersion coefficient [m³/s]
- $x$ is the spatial coordinate [m]
- $t$ is time [s]

**Physical Interpretation:**
- **Nonlinear term** ($\varepsilon u \partial_x u$): Steepens wave profiles, causing wave breaking in the absence of dispersion
- **Dispersion term** ($\mu \partial_x^3 u$): Spreads wave energy, with shorter wavelengths traveling faster
- **Soliton balance**: Exact equilibrium between nonlinearity and dispersion creates stable traveling waves

**Conservation Laws:**

The KdV equation preserves three fundamental invariants:

1. **Mass**: $M = \int_{-\infty}^{\infty} u \, dx$

2. **Momentum**: $P = \int_{-\infty}^{\infty} u^2 \, dx$

3. **Energy**: $E = \int_{-\infty}^{\infty} \left[\frac{\varepsilon}{2} u^3 - \frac{3\mu}{2} \left(\frac{\partial u}{\partial x}\right)^2\right] dx$

These conservation laws serve as critical diagnostic tools for assessing numerical accuracy.

## Numerical Methods

### Spatial Discretization: Pseudo-Spectral Method

The solver employs the Fourier pseudo-spectral method for spatial derivatives, achieving exponential convergence for smooth, periodic solutions.

**Discrete Fourier Transform:**

For a periodic function $u(x)$ discretized on $N$ points $x_j = x_{\text{min}} + j \Delta x$ where $j = 0, 1, \ldots, N-1$ and $\Delta x = (x_{\text{max}} - x_{\text{min}})/N$:

$$\hat{u}_k = \frac{1}{N} \sum_{j=0}^{N-1} u_j e^{-2\pi i k j / N}$$

**Spectral Derivative Computation:**

First derivative:
$$\frac{\partial u}{\partial x}(x_j) = \mathcal{F}^{-1}\left[i k \hat{u}_k\right]$$

Third derivative:
$$\frac{\partial^3 u}{\partial x^3}(x_j) = \mathcal{F}^{-1}\left[(i k)^3 \hat{u}_k\right]$$

where $k$ are the wavenumbers:

$$k_n = \begin{cases}
\frac{2\pi n}{L} & \text{for } n = 0, 1, \ldots, N/2-1 \\
\frac{2\pi (n-N)}{L} & \text{for } n = N/2, \ldots, N-1
\end{cases}$$

with $L = x_{\text{max}} - x_{\text{min}}$ being the domain length.

**Advantages:**
- Exponential accuracy for smooth solutions: error $\sim \mathcal{O}(e^{-cN})$
- Exact derivative computation in Fourier space
- No numerical dissipation or dispersion
- Efficient $\mathcal{O}(N \log N)$ complexity via FFT

### Temporal Integration: DOP853 Method

Time integration uses the Dormand-Prince 8(5,3) method—an explicit Runge-Kutta scheme with adaptive step size control.

**Method Properties:**
- **Order**: 8th-order accurate with 5th and 3rd order embedded error estimates
- **Stages**: 12 function evaluations per step
- **Adaptivity**: Automatic step size adjustment based on local error tolerance

**Semi-Discrete Form:**

After spatial discretization, the KdV equation becomes an ODE system:

$$\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}, t)$$

where:

$$\mathbf{F}(\mathbf{u}, t) = -\varepsilon \mathbf{u} \odot \mathcal{F}^{-1}[i k \hat{\mathbf{u}}] - \mu \mathcal{F}^{-1}[(ik)^3 \hat{\mathbf{u}}]$$

**Error Control:**

The local error at each time step is estimated as:

$$\text{err} = \|\mathbf{u}_8 - \mathbf{u}_5\|$$

where $\mathbf{u}_8$ and $\mathbf{u}_5$ are the 8th and 5th order solutions, respectively.

Step size adjustment:
$$\Delta t_{\text{new}} = 0.9 \Delta t_{\text{old}} \left(\frac{\text{tol}}{\text{err}}\right)^{1/8}$$

**Tolerance Specification:**
- Relative tolerance (rtol): typically $10^{-10}$
- Absolute tolerance (atol): typically $10^{-12}$

Combined error criterion:
$$\text{err} < \text{rtol} \cdot \|\mathbf{u}\| + \text{atol}$$

### Parallel Acceleration

**Numba JIT Compilation:**

Critical computational kernels are compiled using Numba's `@njit(parallel=True)` decorator:

```python
@njit(parallel=True, cache=True)
def compute_nonlinear_term(u, u_x, eps):
    result = np.empty_like(u)
    for i in prange(len(u)):
        result[i] = -eps * u[i] * u_x[i]
    return result
```

**Performance Gains:**
- 10-100× speedup over pure Python/NumPy
- Multi-core parallelization via `prange`
- Automatic vectorization and loop optimization

## Features

- **Spectral accuracy**: Exponential convergence for smooth solutions
- **Adaptive time stepping**: DOP853 (8th-order Runge-Kutta)
- **High-performance computing**: Numba JIT compilation (10-100× speedup)
- **Parallel processing**: Multi-core CPU utilization
- **Professional visualization**: 3D animated GIFs with conservation diagnostics
- **Scientific data format**: NetCDF4 output with CF-1.8 conventions
- **Conservation monitoring**: Real-time tracking of mass, momentum, and energy

## Directory Structure

```
sangkuriang-ideal-solver/
├── configs/                          # Simulation configuration files
│   ├── case1_single_soliton.txt     # Single soliton baseline
│   ├── case2_two_solitons.txt       # Two equal solitons
│   ├── case3_collision.txt          # Soliton collision
│   └── case4_three_solitons.txt     # Three-body system
│
├── src/sangkuriang_ideal/           # Main package source
│   ├── __init__.py                  # Package initialization
│   ├── cli.py                       # Command-line interface
│   │
│   ├── core/                        # Core numerical algorithms
│   │   ├── __init__.py
│   │   ├── solver.py                # KdV solver with spectral methods
│   │   └── initial_conditions.py   # Soliton profile generators
│   │
│   ├── io/                          # Input/output handlers
│   │   ├── __init__.py
│   │   ├── config_manager.py       # Configuration file parser
│   │   └── data_handler.py         # NetCDF writer
│   │
│   ├── visualization/               # Animation and plotting
│   │   ├── __init__.py
│   │   └── animator.py             # 3D GIF generator
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logger.py               # Simulation logging
│       └── timer.py                # Performance profiling
│
├── outputs/                         # Generated results (created at runtime)
│   ├── *.nc                        # NetCDF data files
│   └── *.gif                       # Animation files
│
├── logs/                           # Simulation logs (created at runtime)
│   └── *.log
│
├── pyproject.toml                  # Poetry build configuration
├── README.md                       # This file
├── LICENSE                         # WTFPL license
└── .gitignore                     # Git ignore patterns
```

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

# Initialize solver
solver = KdVSolver(nx=512, x_min=-30.0, x_max=30.0, n_cores=8)

# Create initial condition (sech² profile)
u0 = SechProfile(amplitude=4.0, width=2.0, position=-10.0)(solver.x)

# Solve
result = solver.solve(u0=u0, mu=0.1, eps=0.2, t_final=50.0)

# Check conservation
print(f"Energy error: {result['energy_error']:.2e}")
```

## Test Cases

| Case | Description | Physics | Initial Conditions |
|------|-------------|---------|-------------------|
| 1 | Single soliton | Baseline propagation | $u(x,0) = 4 \operatorname{sech}^2((x+10)/2)$ |
| 2 | Two equal solitons | Phase shift interaction | Two identical solitons |
| 3 | Collision | Overtaking phenomenon | $A_1=6, A_2=2$ (different heights) |
| 4 | Three-soliton system | Complex multi-body dynamics | $A_1=7, A_2=4, A_3=2.5$ |

**Soliton Velocity Formula:**
$$v = \frac{\varepsilon A}{3}$$

where $A$ is the soliton amplitude. Taller solitons travel faster, leading to overtaking collisions.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{herho202x_sangkuriang,
  title   = {sangkuriang: an Idealized {K}orteweg-de {V}ries soliton solver with pseudo-spectral methods},
  author  = {Herho, Sandy H. S. and Khadami, Faruq and Anwar, Iwan P.},
  journal = {xxx},
  volume  = {xxx},
  pages   = {xxx--xxx},
  year    = {202x},
  doi     = {10.xxxx/xxxxx}
}
```

## Authors

- Sandy H.S. Herho
- Faruq Khadami
- Iwan P. Anwar

## License

WTFPL - Do What The F*ck You Want To Public License
